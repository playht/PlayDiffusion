import numpy as np
import soundfile as sf

def make_16bit_pcm(gen):
    """
    return a 16-bit PCM audio numpy ndarray from play_tts vocoder output
    """
    gen_np = gen.cpu().numpy()

    i = np.iinfo("int16")
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max

    if np.isnan(gen_np).any() or np.isinf(gen_np).any():
        raise ValueError("gen_np contains NaN or inf values")

    gen_np = (gen_np * abs_max + offset).clip(i.min, i.max).astype("int16")

    if gen_np.shape[0] != 1:
        raise ValueError(f"gen_np has unexpected shape {gen_np.shape}")

    return gen_np

def save_audio(fname, gen, output_frequency):
    """
    save 16-bit PCM audio file from play_tts vocoder output tensor gen to fname
    """
    gen_np = make_16bit_pcm(gen)
    sf.write(fname, np.squeeze(gen_np[0]), output_frequency, subtype='PCM_16')
