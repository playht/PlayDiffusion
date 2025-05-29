import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from playdiffusion.models.mel_spectrogram.mel import MelSpectrogram
from playdiffusion.utils.voice_resource import VoiceResource
from playdiffusion.models.model_manager import PlayDiffusionModelManager

@torch.inference_mode()
def get_voice_embeddings(
    mm: PlayDiffusionModelManager,
    voice_resource: VoiceResource,
):
    cuda_stream = torch.cuda.Stream()
    with torch.cuda.stream(cuda_stream):
        voice_audio_ar = voice_resource.get_audio(sample_rate=mm.voice_encoder.mel_sample_rate)
        cuda_stream.wait_stream(torch.cuda.current_stream())
        voice_emb  = mm.voice_encoder.get_voice_embedding(voice_audio_ar)

        if voice_emb.isnan().any():
            raise ValueError("NaN in voice_emb")

        vt = mm.vocoder.cond_emb_type
        assert vt == "ar_emb_no_gain"
        result = (voice_emb, voice_emb.to(mm.vocoder.dtype))
        torch.cuda.current_stream().wait_stream(cuda_stream)

    return result

@torch.no_grad()
def get_voice_embedding(
    audio: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    normalize_audio: bool = True,
    clip_duration: float = 30.0,
    uncond_speech: Optional[torch.Tensor] = None,
    mel_sample_rate: int = 24000,
    voice_encoder: nn.Module = None,
    mel: MelSpectrogram = None,
):
    """Compute voice embedding for given speech, or return the unconditional voice embedding if no speech is given.

    Args:
        audio: The speech audio tensor(s) in mel_sample_rate, 30s recommended.
        normalize_audio (bool, optional): Whether to normalize the audio. Defaults to True.
        clip_duration (float, optional): The duration of the audio clip in seconds. Defaults to 30.0.
        uncond_speech: The unconditional voice embedding. (1, 1024)
        mel_sample_rate (int, optional): The sample rate of the mel spectrogram. Defaults to 24000.
        voice_encoder (nn.Module, optional): The voice encoder model. Defaults to None.
        mel (MelSpectrogram, optional): The mel spectrogram model. Defaults to None.

    Returns:
        torch.Tensor: The voice embedding. (B, 1, 1024)
    """
    if audio is None and uncond_speech is not None:
        return uncond_speech.squeeze(0)

    if isinstance(audio, list):
        audio = torch.cat(audio, dim=1)

    clip_size = int(mel_sample_rate * clip_duration)
    voice_encoder = voice_encoder
    if audio.shape[-1] > clip_size:
        # Split long audio into several clips.
        chunk_num = audio.shape[-1] // clip_size
        audio = torch.nn.utils.rnn.pad_sequence(
            [audio[:, i * clip_size: (i + 1) * clip_size] for i in range(chunk_num)],
            batch_first=True,
            padding_value=0)
    elif audio.shape[-1] < clip_size:
        # pad to 30s if necessary.
        audio = F.pad(audio, (0, clip_size - audio.shape[-1]))

    # make sure audio is floating point
    assert torch.is_floating_point(audio), f"Audio must be floating point, got {audio.dtype}"

    if audio.ndim == 2:
        audio = audio[None]

    if audio.shape[1] != 1:
        audio = audio.mean(1, keepdim=True)

    audio = audio.to(voice_encoder.dtype).to(voice_encoder.device)

    embs = []
    for i in range(audio.shape[0]):
        if torch.max(torch.abs(audio[i])) == 0:
            print(f"Zero audio in chunk {i}. Skipping.")
            continue

        if normalize_audio:
            audio[i] = audio[i] / torch.max(torch.abs(audio[i]))

        if audio[i].isnan().any():
            print(f"NaN in audio in chunk {i}. Skipping.")
            continue

        mel_spectogram = mel.encode(audio[i])

        emb = voice_encoder(mel_spectogram)
        if emb.isnan().any():
            print(f"NaN in voice embedding in chunk {i}. Skipping.")
            continue
        embs.append(emb)

    if len(embs) == 0:
        raise ValueError("No valid voice embeddings were computed")

    emb = torch.cat(embs, dim=0)

    if audio.shape[0] > 1:
        emb = torch.mean(emb, dim=0, keepdim=True)

    # (B, 1024) -> (B, 1, 1024)
    emb = emb.unsqueeze(1)

    return emb
