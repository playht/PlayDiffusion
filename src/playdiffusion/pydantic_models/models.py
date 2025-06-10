from typing import Dict, List, Optional

from pydantic import BaseModel, Field

class BaseInput(BaseModel):
    output_text: str = Field(
        description="Text to inpaint",
    )
    num_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of steps to take in the inpainting process",
    )
    init_temp: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Initial temperature for the inpainting process",
    )
    init_diversity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Initial diversity for the inpainting process",
    )
    guidance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Guidance for the inpainting process",
    )
    rescale: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Rescale for the inpainting process",
    )
    topk: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Top-k for the inpainting process",
    )
    audio_token_syllable_ratio: Optional[float] = Field(
        default=None,
        ge=5.0,
        le=25.0,
        description="Ratio of audio tokens to syllables in the input text; if not provided, \
            it will be calculated automatically",
    )


class InpaintInput(BaseInput):
    audio: str = Field(
        description="URL to the audio file to inpaint",
    )
    input_text: str = Field(
        description="Text of the input audio",
    )
    input_word_times: List[Dict] = Field(
        description="Word times of the input audio; each word is a dictionary with the \
            word, start time, and end time",
        examples=[
            [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ]
        ],
        default=[],
    )

class TTSInput(BaseInput):
    voice: str = Field(
        description="URL to the voice resource to use for TTS",
    )

class RVCInput(BaseModel):
    source_speech: str = Field(
        description="URL to the voice resource to use for speech semantics",
    )
    target_voice: str = Field(
        description="URL to the target voice",
    )

