from typing import Dict, List, Literal

from pydantic import BaseModel, Field

FormatType = Literal["raw", "wav", "mp3", "ogg", "flac", "mulaw", "pcm", "wav_mulaw"]

class InpainterBaseInput(BaseModel):
    output_text: str = Field(
        description="Text to inpaint",
    )
    output_format: FormatType = Field(
        default="wav",
        description="Output format of the audio file",
    )
    output_sample_rate: int = Field(
        default=48_000,
        ge=8_000,
        le=48_000,
        description="Output sample rate of the audio file",
    )
    output_speed: float = Field(
        default=1.0,
        gt=0.0,
        le=5.0,
        description="Output speed of the audio file",
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


class InpainterInput(InpainterBaseInput):
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
