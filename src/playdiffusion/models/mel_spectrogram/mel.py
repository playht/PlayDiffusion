import torch
import torchaudio

from playdiffusion.models.mel_spectrogram.tacotron import TacotronSTFT

# Stats from the VqVAE training set - kept so both models use the same MEL format
TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254

UNIV_MEL_MEAN = [ -0.362987, -0.275609, -0.214799, -0.176162, -0.176303, -0.195624, -0.188264, -0.217266, -0.206322, -0.236198, -0.230255, -0.242623, -0.248435, -0.250025, -0.271565, -0.264910, -0.300047, -0.295445, -0.324223, -0.327090, -0.338537, -0.347669, -0.342723, -0.363253, -0.345126, -0.369813, -0.355268, -0.369003, -0.365699, -0.368747, -0.369996, -0.378038, -0.367995, -0.379941, -0.379833, -0.381345, -0.383270, -0.384336, -0.384647, -0.384184, -0.385162, -0.388269, -0.384823, -0.394650, -0.391810, -0.397424, -0.396999, -0.400669, -0.406701, -0.411146, -0.415520, -0.415664, -0.417809, -0.416316, -0.415557, -0.414445, -0.416581, -0.419495, -0.425628, -0.434221, -0.445579, -0.454643, -0.460440, -0.463075, -0.465277, -0.466486, -0.467568, -0.471848, -0.478277, -0.488004, -0.499497, -0.512124, -0.523796, -0.534602, -0.545295, -0.555952, -0.564757, -0.571438, -0.574683, -0.576825, -0.578338, -0.582372, -0.587675, -0.591264, -0.594200, -0.597116, -0.601883, -0.608677, -0.614729, -0.622859, -0.636373, -0.629170, -0.639739, -0.663590, -0.689774, -0.711324, -0.729637, -0.757470, -0.792535, -0.827708 ]
UNIV_MEL_STD = [ 0.484793, 0.558647, 0.617948, 0.650929, 0.653919, 0.645172, 0.652674, 0.635840, 0.642193, 0.624010, 0.627954, 0.620662, 0.618184, 0.617792, 0.604684, 0.606978, 0.583991, 0.584550, 0.566241, 0.563311, 0.555793, 0.549769, 0.551699, 0.538963, 0.548138, 0.532589, 0.539796, 0.531262, 0.532754, 0.531231, 0.530600, 0.526173, 0.532573, 0.526033, 0.525931, 0.524744, 0.523234, 0.521965, 0.521219, 0.521147, 0.519988, 0.517817, 0.519525, 0.513181, 0.514215, 0.510456, 0.510157, 0.507440, 0.503230, 0.499716, 0.496558, 0.496374, 0.495435, 0.496737, 0.497759, 0.499278, 0.498958, 0.497970, 0.494302, 0.488204, 0.479295, 0.471905, 0.466698, 0.464347, 0.463399, 0.463593, 0.463878, 0.462138, 0.458525, 0.451914, 0.443669, 0.434572, 0.426364, 0.418768, 0.411126, 0.403654, 0.398076, 0.395058, 0.394470, 0.394390, 0.393910, 0.391522, 0.387822, 0.386098, 0.385490, 0.384492, 0.380791, 0.375684, 0.373089, 0.368683, 0.359776, 0.373195, 0.379856, 0.377202, 0.370769, 0.364200, 0.355851, 0.338883, 0.313291, 0.283525 ]

def normalize_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1

def diff_normalize_mel(mel):
    nmel = normalize_mel(mel)
    nmel = nmel - torch.tensor(UNIV_MEL_MEAN, device=nmel.device, dtype=nmel.dtype).view(1, -1, 1)
    nmel = nmel / torch.tensor(UNIV_MEL_STD, device=nmel.device, dtype=nmel.dtype).view(1, -1, 1)
    return nmel

class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        mel_fmin=0,
        mel_fmax=12000,
        sampling_rate=24000,
        normalize=False,
        mel_norm_file=None,
        do_diff_normalization=True,
        mel_implementation="tacotron",
    ):
        super().__init__()
        assert mel_implementation in set(['torch', 'tacotron'])
        self.mel_implementation = mel_implementation

        if self.mel_implementation == 'torch':
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                n_fft=filter_length,
                hop_length=hop_length,
                win_length=win_length,
                power=2,
                normalized=normalize,
                sample_rate=sampling_rate,
                f_min=mel_fmin,
                f_max=mel_fmax,
                n_mels=n_mel_channels,
                norm="slaney",
            )
        else:
            self.mel_stft = TacotronSTFT(filter_length,
                hop_length,
                win_length,
                n_mel_channels,
                sampling_rate,
                mel_fmin,
                mel_fmax)
        self.do_diff_normalization = do_diff_normalization
        if mel_norm_file is not None:
            self.mel_norms = torch.load(mel_norm_file)
            self.mel_norms = self.mel_norms.unsqueeze(0).unsqueeze(-1)
        else:
            self.mel_norms = None

    def _apply(self, fn):
        super()._apply(fn)
        if self.mel_norms is not None:
            self.mel_norms = fn(self.mel_norms)
        return self

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor):
        """Encode audio into mel spectrograms.

        Args:
            audio: Tensor of shape (1, T) or (B, 1, T) containing the audio.

        Returns:
            Tensor of shape (1, n_mel_channels, T) or (B, n_mel_channels, T) containing the mel spectrograms.
        """
        return self(audio)

    def forward(self, audio: torch.Tensor):
        if audio.ndim == 3:
            assert audio.shape[1] == 1, f"expected mono audio, got {audio.shape}"
            audio = audio.squeeze(1)
        else:
            assert audio.ndim == 2 and audio.shape[0] == 1, f"expected mono audio, got {audio.shape}"

        mel = self.mel_stft(audio)

        # Perform dynamic range compression
        if self.mel_implementation == 'torch':
            mel = torch.log(torch.clamp(mel, min=1e-5))

        if self.mel_norms is not None:
            mel = mel / self.mel_norms
        elif self.do_diff_normalization:
            mel = diff_normalize_mel(mel)
        return mel