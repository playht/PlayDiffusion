import warnings
from typing import List, Optional, Tuple, Union

import torch
from fairseq2.data import Collater
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device

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
        device: Optional[Device] = None,
        dtype: DataType = torch.float32,
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
    def forward(self, batch: SequenceBatch) -> Tuple[torch.Tensor, PaddingMask]:
        """
        Minimal re-implementation that assumes we only loaded `max_layer` layers.
        This is better as it doesn't require the full model to be loaded.

        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask = self.model.encoder_frontend(batch.seqs, batch.padding_mask)
        encoder_output, padding_mask = self.model.encoder(seqs, padding_mask)
        return encoder_output, padding_mask


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
        dtype: DataType = torch.float16,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__()
        self.collater = Collater(pad_value=1, pad_to_multiple=2)

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

    def create_batch(self, x: BATCH_INPUT) -> SequenceBatch:
        src = self.collater(x)
        seqs, padding_mask = get_seqs_and_padding_mask(src)
        batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask)
        return batch

    @torch.inference_mode()
    def forward(self, batch: SequenceBatch) -> Tuple[torch.Tensor, PaddingMask]:
        self.cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.cuda_stream):
            z, padding_mask = self.encoder(batch)
            units = self.kmeans(z)
            self.gpu_memory_manager.check_and_cleanup()
        torch.cuda.current_stream().wait_stream(self.cuda_stream)
        return units, padding_mask

    @torch.inference_mode()
    def waveform_to_units(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device).to(self.dtype)
        batch = self.create_batch(waveform)
        units, _ = self(batch)
        return units