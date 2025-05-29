import os
import json
from pathlib import Path

import torch
import torchaudio

from playdiffusion.utils.get_resource import get_resource


class VoiceResource:
    @classmethod
    def load(cls, uri, trim_at_sec=None):
        fname = get_resource(uri)
        if fname.endswith(".json"):
            data = json.loads(open(fname, 'r').read())
            sample_paths = data.get("samples")
            sample_paths = [get_resource(s) for s in sample_paths]
            name = data.get("name")
            if name is None:
                name = Path(fname).parent.name
        else:
            # assume uri is a single sample
            sample_paths = [fname]
            name = Path(fname).stem

        return cls(name, sample_paths, None, trim_at_sec)

    @classmethod
    def with_audio(cls, name, samples: list, trim_at_sec=None):
        return cls(name, None, samples, trim_at_sec)

    def __init__(self, name, sample_paths, sample_audios, trim_at_sec):
        self.name = name
        self.sample_paths = sample_paths
        self.sample_audios = sample_audios
        self.audio_per_sr = {}
        self.trim_at_sec = trim_at_sec

    @staticmethod
    def _maybe_resample(audio_tuple, sr):
        audio, orig_sr = audio_tuple
        if sr is not None and orig_sr != sr:
            audio = torchaudio.transforms.Resample(orig_sr, sr)(audio)
        else:
            sr = orig_sr
        return (audio, sr)

    @staticmethod
    def _load_sample(path: str, sr=None):
        audio = torchaudio.load(path)
        return VoiceResource._maybe_resample(audio, sr)

    def _get_sample_audio(self):
        if self.sample_audios:
            return self.sample_audios
        if self.sample_paths is None:
            raise ValueError(f"can't load samples with no paths")
        self.sample_audios = [self._load_sample(path) for path in self.sample_paths]
        return self.sample_audios

    def get_audio(self, sample_rate):
        cached = self.audio_per_sr.get(sample_rate)
        if cached is not None:
            return cached

        rsamples = [self._maybe_resample(audio, sample_rate)[0] for audio in self._get_sample_audio()]
        silence = torch.zeros(1, int(sample_rate * 0.25))
        samples = []
        for s in rsamples:
            samples.append(s)
            # add some silence inbetween
            samples.append(silence)
        samples.pop()

        audio = torch.cat(samples, dim=1)

        if self.trim_at_sec is not None and audio.shape[1] > sample_rate * self.trim_at_sec:
            audio = audio[:, : sample_rate * self.trim_at_sec]

        self.audio_per_sr[sample_rate] = audio
        return audio

    # backwards compatibility
    def load_audio(self, sample_rate=24000, trim_at_sec=None):
        if trim_at_sec != self.trim_at_sec:
            raise ValueError(f"trim at sec mismatch")
        return self.get_audio(sample_rate)

    def save(self, prefix = None, with_manifest = False, flat = False):
        if prefix:
            os.makedirs(prefix, exist_ok = True)
            base_path = prefix + '/'
        else:
            base_path = ''
        base_path += self.name

        if flat:
            if len(self._get_sample_audio()) > 1:
                base_path += '_'
        else:
            base_path += '/'
            os.makedirs(base_path, exist_ok = True)

        saved = []
        for i, audio in enumerate(self._get_sample_audio()):
            if i == 0:
                path = f"{base_path}.wav" if flat else f"{base_path[:-1]}.wav"
            else:
                path = f"{base_path}{i}.wav"
            torchaudio.save(path, *audio)
            saved.append(path)

        if with_manifest:
            with open(base_path + 'manifest.json', 'w') as f:
                json.dump({'name': self.name, 'samples': saved}, f)

        return saved
