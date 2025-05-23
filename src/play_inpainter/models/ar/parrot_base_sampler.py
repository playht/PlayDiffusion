from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from play_inpainter.models.ar.parrot_base_model import BaseModel
from play_inpainter.models.mel_spectrogram.mel import MelSpectrogram
from play_inpainter.utils.voice_emb import get_voice_embedding

class BaseSampler(nn.Module):
    @dataclass
    class SamplerParams:
        num_samples: int = 1
        # Higher values will create a better match to the input voice, but may also cause text to be skipped or mispronounced.
        voice_guidance: float = 0.3
        # Higher values will create a better match to the input style, but may also cause text to be skipped or mispronounced.
        style_guidance: float = 0.1
        # Higher values will create a better match to the input text, but may cause the speech to sound less natural.
        text_guidance: float = 0.5
        # Higher values will create more diverse speech, too high will cause artifacts.
        temperature: float = 0.2
        # Higher values will create more diverse speech, too high will cause artifacts.
        top_p: float = 0.5
        len_penalty: float = 1.1
        # Lower values will cause more repetition (pauses, words, phrases), too low will cause artifacts or infinite pauses.
        repetition_penalty: float = 1.10
        # Style tags, set to None is unconditional
        language_identifier: Optional[str] = None
        audio_source: Optional[str] = None
        speaker_attributes: Optional[str] = None
        speech_attributes: Optional[str] = None
        repetition_penalty_window: int = 30
        recency_penalty_window: int = 5
        num_text_pad_tokens: int = 30
        num_short_text_pad_tokens: int = 20
        max_mel_seq_len: int = 1024
        language_identifier_v3: Optional[str] = None

    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model
        self.mel = MelSpectrogram.load_preset(f"voice_conditioning_{model.version}")

    @property
    def device(self):
        return self.model.device

    @property
    def mel_sample_rate(self):
        return self.model.mel_sample_rate

    @property
    def voice_encoder(self):
        return self.model.voice_encoder

    @property
    def voice_encoder_gain(self):
        return self.model.voice_encoder_gain

    @property
    def voice_encoder_dtype(self):
        return self.model.voice_encoder_gain.dtype

    @torch.inference_mode()
    def get_voice_embedding_and_gain(
        self,
        audio: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        normalize_audio: bool = True,
        clip_duration: float = 30.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_voice_embedding(
            audio,
            normalize_audio=normalize_audio,
            apply_gain=False,
            clip_duration=clip_duration,
            uncond_speech=self.model.uncond_speech,
            mel_sample_rate=self.mel_sample_rate,
            voice_encoder=self.voice_encoder,
            voice_encoder_gain=self.voice_encoder_gain,
            mel=self.mel
        ), self.voice_encoder_gain
