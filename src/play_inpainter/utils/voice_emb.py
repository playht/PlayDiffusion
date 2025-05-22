import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from play_inpainter.models.mel_spectrogram.mel import MelSpectrogram
from play_inpainter.utils.voice_resource import VoiceResource
from play_inpainter.models.model_manager import InpainterModelManager

def get_voice_embeddings(
    mm: InpainterModelManager,
    voice_resource: VoiceResource,
    normalize_audio: bool = True,
    clip_duration: float = 30.0,
):
    cuda_stream = torch.cuda.Stream()
    with torch.cuda.stream(cuda_stream):
        voice_audio_ar = voice_resource.get_audio(sample_rate=mm.mel_sample_rate)
        cuda_stream.wait_stream(torch.cuda.current_stream())
        voice_emb_ng = get_voice_embedding(
            voice_audio_ar,
            normalize_audio=normalize_audio,
            apply_gain=False,
            clip_duration=clip_duration,
            uncond_speech=mm.uncond_speech,
            mel_sample_rate=mm.mel_sample_rate,
            voice_encoder=mm.voice_encoder,
            voice_encoder_gain=mm.voice_encoder_gain,
            mel=mm.mel
        )

        voice_emb = apply_voice_embedding_gain(voice_emb_ng, mm.voice_encoder_gain)

        if voice_emb.isnan().any():
            raise ValueError("NaN in voice_emb")

        vt = mm.vocoder.cond_emb_type
        if vt == 'ar_emb_with_gain':
            result = (voice_emb, voice_emb.to(mm.vocoder.dtype))
        elif vt == 'ar_emb_no_gain':
            result = (voice_emb, voice_emb_ng.to(mm.vocoder.dtype))
        elif vt == 'wav@24khz':
            voice_audio_vc = voice_resource.get_audio(sample_rate=24000)
            vocoder_emb = mm.vocoder.get_cond_emb(voice_audio_vc, None, None)
            result = (voice_emb, vocoder_emb)
        else:
            raise ValueError(f"Unknown vocoder cond_emb_type: '{vt}'")

        torch.cuda.current_stream().wait_stream(cuda_stream)

    return result

def apply_voice_embedding_gain(emb: torch.Tensor, voice_encoder_gain):
    return emb * voice_encoder_gain


@torch.inference_mode()
def get_voice_embedding(
    audio: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    normalize_audio: bool = True,
    apply_gain: bool = True,
    clip_duration: float = 30.0,
    uncond_speech: Optional[torch.Tensor] = None,
    mel_sample_rate: int = 24000,
    voice_encoder: nn.Module = None,
    voice_encoder_gain: torch.Tensor = None,
    mel: MelSpectrogram = None,
):
    """Compute voice embedding for given speech, or return the unconditional voice embedding if no speech is given.

    Args:
        audio: The speech audio tensor(s) in mel_sample_rate, 30s recommended.
        normalize_audio (bool, optional): Whether to normalize the audio. Defaults to True.
        apply_gain (bool, optional): Whether to apply the voice encoder gain. Defaults to True.
        clip_duration (float, optional): The duration of the audio clip in seconds. Defaults to 30.0.
        uncond_speech: The unconditional voice embedding. (1, 1024)
        mel_sample_rate (int, optional): The sample rate of the mel spectrogram. Defaults to 24000.
        voice_encoder (nn.Module, optional): The voice encoder model. Defaults to None.
        voice_encoder_gain (torch.Tensor, optional): The voice encoder gain. Defaults to None.
        mel (MelSpectrogram, optional): The mel spectrogram model. Defaults to None.

    Returns:
        torch.Tensor: The voice embedding. (B, 1, 1024)
    """
    if audio is None:
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

    audio = audio.to(voice_encoder_gain.dtype).to(voice_encoder_gain.device)

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

    if apply_gain:
        emb = apply_voice_embedding_gain(emb, voice_encoder_gain)

    return emb
