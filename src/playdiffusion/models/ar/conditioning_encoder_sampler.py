from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from playdiffusion.models.mel_spectrogram.mel import MelSpectrogram
from playdiffusion.utils.voice_emb import get_voice_embedding

class ConditioningEncoderSampler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mel = MelSpectrogram()

    @property
    def device(self):
        return self.model.device

    @property
    def mel_sample_rate(self):
        return self.model.mel_sample_rate

    @torch.inference_mode()
    def get_voice_embedding(
        self,
        audio: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        normalize_audio: bool = True,
        clip_duration: float = 30.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_voice_embedding(
            audio,
            normalize_audio=normalize_audio,
            clip_duration=clip_duration,
            mel_sample_rate=self.mel_sample_rate,
            voice_encoder=self.model,
            mel=self.mel
        )
