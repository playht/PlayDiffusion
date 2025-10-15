import warnings
from typing import List, Optional, Tuple, Union

import torch
from fairseq2.nn import BatchLayout
from torch.nn.utils.rnn import pad_sequence

from playdiffusion.models.speech_tokenizer.kmeans import KmeansModel
from playdiffusion.models.speech_tokenizer.xlsr_encoder import load_xlsr_encoder
from playdiffusion.utils.gpu_memory_manager import GPUMemoryManager


BATCH_INPUT = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]

# Disable "weight_norm" deprecation wanring for loading the Wav2Vec Model
warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

class SpeechEncoder(torch.nn.Module):
    """
    A wrapper Module for the XLS-R 1B model that only loads the first `max_layer` layers.

    Extracts the intermediate representations from the model.
    Waveform -> Layer 35 latents
    """

    def __init__(
        self,
        checkpoint: Union[str, None] = "data/checkpoints/xlsr2_1b_v2_custom.pt",
        max_layer: Union[int, None] = 35,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        strict: bool = False,
        eval: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.layer = max_layer
        self.layer_idx = None if max_layer is None else max_layer - 1

        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Initializes the Encoder but only until the `max_layer`
        # We don't need the full model for the intermediate representations
        self.model, self.config, self.encoder_config = load_xlsr_encoder(
            device=device, dtype=dtype, max_layer=max_layer
        )

        # Load checkpoint
        if checkpoint is not None:
            sd = torch.load(checkpoint)
            # Using strict=False because we can't load the
            if max_layer is not None:
                strict = False
            # remaining layers that don't exist when using the `max_layer`
            self.model.load_state_dict(sd, strict=strict)

        # The hook approach prevents the last model.encoder.layer_norm to be applied!
        # As this layernorm is trained to be applied on the final layer representations
        # we should NOT apply it for the intermediate representation!
        # Therefore the layernorm is set to NONE when we load the smaller model.
        if max_layer is not None:
            self.model.encoder.layer_norm = None  # type: ignore

        if eval:
            # Set to evaluation (no dropout, loss calc in forward pass)
            self.model.eval()
            self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.inference_mode()
    # The forward signature now accepts the padded sequences and the batch layout
    def forward(self, seqs: torch.Tensor, layout: BatchLayout) -> torch.Tensor:
        """
        Minimal re-implementation that assumes we only loaded `max_layer` layers.
        This is better as it doesn't require the full model to be loaded.

        :param seqs:
            The batch of padded sequences.
        :param layout:
            The layout of the batch (containing sequence lengths).
        """
        seqs, layout_out = self.model.encoder_frontend(seqs, layout)
        encoder_output = self.model.encoder(seqs, layout_out)
        return encoder_output, layout_out


class SpeechTokenizer(torch.nn.Module):
    """
    A wrapper Module for the XLS-R 1B Encoder together with the
    pre-trained kmeans discretization layer.

    Extracts Units (i.e., Audio-Units) from the input waveform.

    Waveform -> Layer 35 latents -> Kmeans -> Units

    The units are simply the indices of the "codebook"
    defined by the kmeans model (ie., 10K num_embeddings).
    """

    def __init__(
        self,
        checkpoint: Union[str, None] = "data/checkpoints/xlsr2_1b_v2_custom.pt",
        kmeans_layer_checkpoint: str = "data/checkpoints/kmeans_10k.npy",
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.encoder = SpeechEncoder(dtype=dtype, device=device, checkpoint=checkpoint, max_layer=35)
        self.kmeans = KmeansModel(kmeans_layer_checkpoint, device=self.encoder.device, dtype=self.encoder.dtype)
        self.gpu_memory_manager = GPUMemoryManager(threshold_percent=85, min_interval_seconds=1)
        self.cuda_stream = torch.cuda.Stream()


    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def create_batch(self, x: BATCH_INPUT) -> Tuple[torch.Tensor, BatchLayout]:
        if isinstance(x, torch.Tensor):
            x = [x]
        
        lens: List[int] = [int(t.shape[0]) for t in x]
        # Original code padded with 1, but for an audio model 0 makes more sense
        seqs = pad_sequence(x, batch_first=True, padding_value=0.0)
        seqs = seqs.to(self.device, self.dtype)

        B, T_max = int(seqs.size(0)), int(seqs.size(1))
        seqs_layout = BatchLayout(shape=(B, T_max), seq_lens=lens, device=seqs.device)
        return seqs, seqs_layout

    @torch.inference_mode()
    def forward(self, seqs: torch.Tensor, seqs_layout: BatchLayout) -> tuple[torch.Tensor, BatchLayout]:
        
        units = None
        if torch.cuda.is_available():
            self.cuda_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cuda_stream):
                z, unit_layout = self.encoder(seqs, seqs_layout)
                units = self.kmeans(z)
                self.gpu_memory_manager.check_and_cleanup()
            torch.cuda.current_stream().wait_stream(self.cuda_stream)
        else:
            z, unit_layout = self.encoder(seqs, seqs_layout)
            units = self.kmeans(z) # Doesn't modify layout
        return units, unit_layout

    @torch.inference_mode()
    def waveform_to_units(self, waveform: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, BatchLayout]:
        """
        Converts a single waveform tensors or a list of waveform tensors into audio tokens.
        
        Returns a batch of audio tokens [B, T] and the corresponding BatchLayout 
        Use unit_layout.seq_lens to get the length of the individual audio token tensors.

        Output units are tokens of dtype torch.int64
        0 <= token < num_embeddings (e.g., 10000 for kmeans_10k.npy)
        """
        seqs, seqs_layout = self.create_batch(waveform)
        units, unit_layout = self(seqs, seqs_layout)
        return units, unit_layout