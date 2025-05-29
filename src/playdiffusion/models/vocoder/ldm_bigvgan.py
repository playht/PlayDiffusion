from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional
from torch import pow, sin
from torch.nn import Conv1d, ConvTranspose1d, Parameter, Linear
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm
from playdiffusion.utils.gpu_memory_manager import GPUMemoryManager

class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

def load_ldm_bigvgan(
        checkpoint: str,
        kmeans_layer_checkpoint: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda")
    ) -> BigVGAN:
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    h = checkpoint.get("params")
    state = checkpoint["generator"]

    h = DotDict(h)
    if kmeans_layer_checkpoint is not None:
        h["xlsr_centroids"] = kmeans_layer_checkpoint

    for lib in torch.classes.loaded_libraries:
        if "bigvgan_kernels" in lib:
            h["bigvgan_kernels"] = torch.classes.BigVGAN.Kernels()
            break

    model = BigVGAN(h)
    model.load_state_dict(state)  # type: ignore
    model.eval()
    model.remove_weight_norm()
    model = model.to(device, dtype)

    return model

class Snake(nn.Module):
    '''
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        '''
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

LRELU_SLOPE = 0.1

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    #input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1),
                       stride=self.stride, groups=C)

        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


class FusedAntiAliasActivation(torch.autograd.Function):
    """
    Assumes filter size 12, replication padding on upsampling/downsampling, and logscale alpha/beta parameters as inputs.
    The hyperparameters are hard-coded in the kernel to maximize speed.
    NOTE: The fused kenrel is incorrect for Activation1d with different hyperparameters.
    """

    @staticmethod
    def forward(ctx, bigvgan_kernels, inputs, up_ftr, down_ftr, alpha, beta):
        activation_results = bigvgan_kernels.forward(
            inputs, up_ftr, down_ftr, alpha, beta
        )

        return activation_results

    @staticmethod
    def backward(ctx, output_grads):
        raise NotImplementedError
        return output_grads, None, None


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        bigvgan_kernels = None,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)
        self.bigvgan_kernels = bigvgan_kernels

    # x: [B,C,T]
    def forward(self, x):
        if self.bigvgan_kernels is None:
            x = self.upsample(x)
            x = self.act(x)
            x = self.downsample(x)
            return x
        else:
            if self.act.__class__.__name__ == "Snake":
                beta = self.act.alpha.data  # Snake uses same params for alpha and beta
            else:
                beta = (
                    self.act.beta.data
                )  # Snakebeta uses different params for alpha and beta
            alpha = self.act.alpha.data
            if (
                not self.act.alpha_logscale
            ):  # Exp baked into cuda kernel, cancel it out with a log
                alpha = torch.log(alpha)
                beta = torch.log(beta)

            x = FusedAntiAliasActivation.apply(
                self.bigvgan_kernels, x, self.upsample.filter, self.downsample.lowpass.filter, alpha, beta
            )
            return x

class FiLMLike(torch.nn.Module):
    def __init__(self, h):
        super(FiLMLike, self).__init__()
        self.gamma_nn = nn.Sequential(
            Linear(in_features=h.speaker_cond_dim,
                   out_features=h.ar_tokens_dim//2 if h.no_mel_simulation else h.num_mels
                   ),
        )
        self.beta_nn = nn.Sequential(
            Linear(in_features=h.speaker_cond_dim,
                   out_features=h.ar_tokens_dim//2 if h.no_mel_simulation else h.num_mels
                   ),
        )

    def forward(self, s):
        gamma = self.gamma_nn(s)
        beta = self.beta_nn(s)

        return gamma, beta



class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class ResBlock3(torch.nn.Module):
    def __init__(self,
                 channels,
                 kernel_size=3,
                 dilation=(1, 3),
                 activation='snakebeta',
                 snake_logscale=True,
                 bigvgan_kernels=None):
        super(ResBlock3, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)
        if activation == 'snake':
            self.activation = Activation1d(activation=Snake(channels, alpha_logscale=snake_logscale), bigvgan_kernels=bigvgan_kernels)
        elif activation == 'snakebeta':
            self.activation = Activation1d(activation=SnakeBeta(channels, alpha_logscale=snake_logscale), bigvgan_kernels=bigvgan_kernels)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        for c in self.convs:
            xt = self.activation(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Upsampler_mrf(nn.Module):
    def __init__(self,
            h,
            upsample_initial_channel=1024,
            resblock_kernel_sizes=[3,5,7],
            resblock_dilation_sizes=[[1,2], [2,6], [3,12]],
            use_leaky_relu=True,
            bigvgan_kernels=None):
        super(Upsampler_mrf, self).__init__()
        self.in_channels = h.ar_tokens_dim
        self.mel_channel = h.num_mels
        self.num_upsamples = h.ar_tokens_hop_size // (2 * h.hop_size)
        self.num_kernels = len(resblock_kernel_sizes)

        resblock = ResBlock3 if h.use_snake_pre_processing else ResBlock2

        stride = 2
        in_channels = self.in_channels

        self.ups = nn.ModuleList()
        for i in range(self.num_upsamples):
            self.ups.append(
                weight_norm(ConvTranspose1d(
                    in_channels//(2**i),
                    in_channels//(2**(i+1)),
                    2*stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                if h.use_snake_pre_processing:
                    self.resblocks.append(resblock(ch, k, d, bigvgan_kernels=bigvgan_kernels))
                else:
                    self.resblocks.append(resblock(ch, k, d))

        c1 = in_channels // (2 ** (self.num_upsamples) )
        c2 = in_channels // (2 ** (self.num_upsamples + 1) )

        if h.no_mel_simulation:
            self.post = nn.ModuleList()
            self.post.append(weight_norm(
                    Conv1d(c1, c1, 7, 1, padding=3)))
            self.post.append(weight_norm(
                    Conv1d(c1, c1, 7, 1, padding=3)))
        else:
            self.post = nn.ModuleList()
            self.post.append(weight_norm(
                    Conv1d(c1, c2, 7, 1, padding=3)))
            self.post.append(weight_norm(
                    Conv1d(c2, self.mel_channel, 7, 1, padding=3)))

        # whether to use LeakyReLU at the end of each process
        self.use_leaky = use_leaky_relu

    def forward(self, x):
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            if self.use_leaky:
                x = F.leaky_relu(x, LRELU_SLOPE)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        if self.use_leaky:
            x = F.leaky_relu(x, LRELU_SLOPE)
        for post in self.post:
            x = post(x)
            if self.use_leaky:
                x = F.leaky_relu(x, LRELU_SLOPE)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.post:
            remove_weight_norm(l)


class Upsampler_simple(nn.Module):
    def __init__(self, h):
        super(Upsampler_simple, self).__init__()

        num_upsamples = h.ar_tokens_hop_size // (2*h.hop_size)
        scale_factor = 2 ** num_upsamples

        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.projection = nn.Linear(in_features=h.ar_tokens_dim, out_features=h.num_mels)

    def forward(self, x):
        x = self.up(x)
        x = self.projection(x.permute(0,2,1)).permute(0,2,1)

        return x


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None, bigvgan_kernels=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        if activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale), bigvgan_kernels=bigvgan_kernels)
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale), bigvgan_kernels=bigvgan_kernels)
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "leaky_relu":
            self.activations = nn.ModuleList([nn.LeakyReLU(LRELU_SLOPE) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class LookupTable(torch.nn.Module):
    def __init__(self, h):
        super(LookupTable, self).__init__()
        centroids = torch.from_numpy(np.load(h.xlsr_centroids))
        self.register_buffer('centroids', centroids)
        self.projection = nn.Linear(in_features=self.centroids.size(1), out_features=h.ar_tokens_dim)
        if h.codes_hop_size != h.ar_tokens_hop_size:
           self.scale_factor = h.codes_hop_size / h.ar_tokens_hop_size
        else:
           self.scale_factor = None

    def forward(self, c):
        x = self.centroids[c]
        x = self.projection(x).permute(0,2,1)
        if self.scale_factor is not None:
            return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x

class BigVGAN(torch.nn.Module):
    def __init__(self, h):
        super(BigVGAN, self).__init__()
        self.h = h

        self._output_frequency = h['sampling_rate']
        self._ar_tokens_hop_size = h.ar_tokens_hop_size

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.bigvgan_kernels = h.get('bigvgan_kernels', None)
        self.gpu_memory_manager = GPUMemoryManager(threshold_percent=93, min_interval_seconds=5)

        if 'no_mel_simulation' not in h:
            h['no_mel_simulation'] = False
        if 'use_snake_pre_processing' not in h:
            h['use_snake_pre_processing'] = False

        # pre conv
        if h.no_mel_simulation:
            self.conv_pre = weight_norm(Conv1d(h.ar_tokens_dim // 2, h.upsample_initial_channel, 7, 1, padding=3))
        else:
            self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k,
                                            u,
                                            # padding=(k - u) // 2))
                                            # IMPORTANT NOTE: this format here allows for odd upsampling rates
                                            # without increasing the kernel. It is the approach used in UnivNet
                                            padding=u // 2 + u % 2,
                                            output_padding=u % 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        ch = 0
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(h, ch, k, d, activation=h.activation, bigvgan_kernels=self.bigvgan_kernels))

        # post conv
        if h.activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post, bigvgan_kernels=self.bigvgan_kernels)
        elif h.activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post, bigvgan_kernels=self.bigvgan_kernels)
        elif h.activation == "leaky_relu":
            self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        if h.get("add_upsampler", True):
            self.upsampler = Upsampler_mrf(
                h,
                upsample_initial_channel=h.ar_tokens_dim,
                use_leaky_relu=h.get("add_leaky_relu_in_mrf", True),
                bigvgan_kernels=self.bigvgan_kernels
            ) if h.get("use_mrf", False) else Upsampler_simple(h)
        else:
            self.upsampler = None

        self.cond_layer = FiLMLike(h) if h.get("use_ar_speaker_cond", False) else None
        self.scaler = nn.Parameter(torch.full((1, 1), fill_value=.02)) if h.get("use_scaler", False) else None
        self.lookup_table = LookupTable(h)
        # whether to use TANH or not at the output (from BIGVGAN-V2)
        self.use_tanh_final = h.get("use_tanh_at_final", True)

    @torch.inference_mode()
    def forward(self, x, s):
        x = self.lookup_table(x)

        # apply learnable scaler
        if self.scaler is not None:
            x = x * self.scaler

        if self.upsampler:
            # upsample from 1024 hop_size to 256 hop_size
            x = self.upsampler(x)

        if self.cond_layer:
            # a simplified version of FiLM
            gamma, beta = self.cond_layer(s)

            # applies conditioning to the upsampled feature
            x = x * gamma.unsqueeze(2) + beta.unsqueeze(2)

        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def warmup(self, size_chunk):
        if self.h.from_codes:
            self.sample(
                None,
                torch.randint(1, 1000, (1, size_chunk), dtype=torch.int32, device=self.device),
                torch.randn(1, 1024,  dtype = self.dtype, device=self.device))
        else:
            self.sample(
                None,
                torch.randn(1, self.h.ar_tokens_dim, size_chunk, dtype = self.dtype, device=self.device),
                torch.randn(1, 1024,  dtype = self.dtype, device=self.device))

    @torch.inference_mode()
    def sample(self, text, ar_h, voice_emb):
        return self.sample_impl(ar_h, voice_emb)

    @torch.inference_mode()
    def sample_impl(self, ar_h, voice_emb):
        self.gpu_memory_manager.check_and_cleanup()
        t_start = time.perf_counter()
        if voice_emb.ndim == 3:
            voice_emb = voice_emb.squeeze(1)
        res = self.forward(ar_h, voice_emb)
        print(f"Vocoder time: {(time.perf_counter() - t_start) * 1000:.1f} ms")
        return res

    @property
    def output_frequency(self):
        return self._output_frequency

    @property
    def cond_emb_type(self):
        return "ar_emb_no_gain"

    @property
    def is_diffusion(self):
        return False

    @property
    def device(self):
        return self.conv_pre.weight.device

    @property
    def dtype(self):
        return self.conv_pre.weight.dtype
