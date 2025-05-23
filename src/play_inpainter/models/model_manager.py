class InpainterModelManager:
    def __init__(self, preset: dict, device):
        import torch
        from torch import nn
        from play_inpainter.models.vocoder.ldm_bigvgan import BigVGAN

        self.preset = preset
        self.ar_kwargs = preset["ar"]
        self.vocoder_kwargs = preset["vocoder"]
        self.tokenizer_kwargs = preset["tokenizer"]
        self.speech_tokenizer_kwargs = preset["speech_tokenizer"]
        self.mel_kwargs = preset["mel"]
        self.inpainter_kwargs = preset["inpainter"]
        self.device = device

        self.speech_tokenizer_sample_rate = self.speech_tokenizer_kwargs.pop("sample_rate")
        self.tokenizer, self.speech_tokenizer = (
            self.load_tokenizers(
                tokenizer_config=self.tokenizer_kwargs,
                speech_tokenizer_config=self.speech_tokenizer_kwargs,
            )
        )

        self.mel_sample_rate = self.mel_kwargs["sample_rate"]

        self.ar = self.load_ar(self.ar_kwargs)

        self.vocoder: BigVGAN = self.load_vocoder(self.vocoder_kwargs)

        self.inpainter = self.load_inpainter(self.inpainter_kwargs)

    def load_vocoder(self, config: dict):
        from play_inpainter.models.vocoder.ldm_bigvgan import load_ldm_bigvgan

        print(f"Using vocoder checkpoint: {config['checkpoint']}")

        return load_ldm_bigvgan(
            **config,
            device=self.device,
        )

    def load_tokenizers(
        self,
        tokenizer_config: dict,
        speech_tokenizer_config: dict,
    ):
        from play_inpainter.models.speech_tokenizer.speech_tokenizer import SpeechTokenizer
        from play_inpainter.models.tokenizer.pp_tokenizer import PPTokenizer

        print("Loading tokenizer")
        tokenizer = PPTokenizer(
            **tokenizer_config, device=self.device,
        )

        print("Loading speech tokenizer")
        speech_tokenizer = SpeechTokenizer(
            **speech_tokenizer_config, device=self.device
        )

        return tokenizer, speech_tokenizer

    def load_ar(self, config: dict):
        from play_inpainter.models.ar.parrot_base_model import BaseModel
        from play_inpainter.models.ar.parrot_base_sampler import BaseSampler

        model = BaseModel(**config)
        sampler = BaseSampler(model)

        return sampler

    def load_mel(self):
        from play_inpainter.models.mel_spectrogram.mel import MelSpectrogram

        return MelSpectrogram.load_preset()

    def load_inpainter(self, config: dict):
        from play_inpainter.models.inpainter.masklm_text import load_maskgct_inpainter

        print(f"Using inpainter checkpoint: {config['checkpoint']}")
        return load_maskgct_inpainter(**config, device=self.device)
