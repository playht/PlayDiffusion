import time
import torch

def load_audio_from_file(file_name: str):
    import soundfile as sf

    try:
        data, sample_rate = sf.read(file_name, dtype="float32")
        return sample_rate, data
    except Exception as e:
        raise ValueError(f"Cannot load audio from file `{file_name}`: {e}")


def get_normalization_factor(audio_data):
    """
    Determine the appropriate normalization factor
    based on the audio data type and range.
    """
    import numpy as np

    dtype = audio_data.dtype
    if np.issubdtype(dtype, np.floating):
        # If already float, check if normalization needed
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            return max_abs
        return 1.0
    # For integer types, use the full range of the data type
    info = np.iinfo(dtype)
    return -info.min  # Using negative min because abs(min) > max for int types


def raw_audio_to_torch_audio(raw_audio_np):
    import torch

    sr, audio_data = raw_audio_np
    norm_factor = get_normalization_factor(audio_data)
    torch_audio = torch.from_numpy(audio_data).float() / norm_factor

    if torch_audio.ndim == 1:
        torch_audio = torch_audio.unsqueeze(0)
    else:
        torch_audio = torch_audio.transpose(0, 1)
        torch_audio = torch_audio.mean(dim=0, keepdim=True)  # Convert to mono

    if sr < 16000:
        raise Exception(
            "Garbage in, garbage out. Please provide audio with a sample "
            "rate of at least 16kHz, ideally 24kHz."
        )

    return sr, torch_audio


@torch.inference_mode()
def get_vocoder_embedding(voice_name: str, mm):
    import playdiffusion.utils.voice_emb as voice_emb_util
    from playdiffusion.utils.voice_resource import VoiceResource

    try:
        voice_resource = VoiceResource.load(voice_name)
    except Exception:
        print("Failed to load voice resource")
        raise

    _, vocoder_emb = voice_emb_util.get_voice_embeddings(mm, voice_resource)
    if len(vocoder_emb.shape) > 2:
        vocoder_emb = vocoder_emb.squeeze(0)

    return vocoder_emb


def load_audio(audio_path: str, device):
    from playdiffusion.utils.get_resource import get_resource

    local_audio_path = get_resource(audio_path)
    raw_audio = load_audio_from_file(local_audio_path)
    sr, torch_audio = raw_audio_to_torch_audio(raw_audio)
    torch_audio = torch_audio.to(device)
    print(
        f"Got raw audio: duration {torch_audio.shape[-1] / sr:.3f} s \
            ({sr} Hz, {torch_audio.shape} samples)"
    )

    return torch_audio, sr


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_time = time.time()
        self.times = {}

    def __call__(self, description: str):
        if description in self.times:
            print(f"Already timed {description}")
            return
        new_time = time.time()
        self.times[description] = 1000 * (new_time - self.previous_time)
        print(f"{description} time: {self.times[description]:.1f} ms")
        self.previous_time = new_time

    def get_times(self):
        return self.times
