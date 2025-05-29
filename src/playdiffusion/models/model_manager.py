class PlayDiffusionModelManager:
    def __init__(self, preset: dict, device):
        import torch
        from torch import nn
        from playdiffusion.models.vocoder.ldm_bigvgan import BigVGAN

        self.preset = preset
        self.voice_encoder_kwargs = preset["voice_encoder"]
        self.vocoder_kwargs = preset["vocoder"]
        self.tokenizer_kwargs = preset["tokenizer"]
        self.speech_tokenizer_kwargs = preset["speech_tokenizer"]
        self.inpainter_kwargs = preset["inpainter"]
        self.device = device

        self.speech_tokenizer_sample_rate = self.speech_tokenizer_kwargs.pop("sample_rate")
        self.tokenizer, self.speech_tokenizer = (
            self.load_tokenizers(
                tokenizer_config=self.tokenizer_kwargs,
                speech_tokenizer_config=self.speech_tokenizer_kwargs,
            )
        )

        self.voice_encoder = self.load_voice_encoder(self.voice_encoder_kwargs)

        self.vocoder: BigVGAN = self.load_vocoder(self.vocoder_kwargs)

        self.inpainter = self.load_inpainter(self.inpainter_kwargs)

    def load_vocoder(self, config: dict):
        from playdiffusion.models.vocoder.ldm_bigvgan import load_ldm_bigvgan

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
        from playdiffusion.models.speech_tokenizer.speech_tokenizer import SpeechTokenizer
        from playdiffusion.models.tokenizer.pp_tokenizer import PPTokenizer

        print("Loading tokenizer")
        tokenizer = PPTokenizer(
            **tokenizer_config, device=self.device,
        )

        print("Loading speech tokenizer")
        speech_tokenizer = SpeechTokenizer(
            **speech_tokenizer_config, device=self.device
        )

        return tokenizer, speech_tokenizer

    def load_voice_encoder(self, config: dict):
        from playdiffusion.models.ar.conditioning_encoder import ConditioningEncoder
        from playdiffusion.models.ar.conditioning_encoder_sampler import ConditioningEncoderSampler
        import torch

        saved_dict = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)

        config = saved_dict['config']
        voice_encoder = ConditioningEncoder(
            config['mel_dim'],
            config['model_dim'],
            config['voice_encoder_depth']
        )

        voice_encoder.load_state_dict(saved_dict['model_state_dict'])

        sampler = ConditioningEncoderSampler(voice_encoder)
        sampler.model = sampler.model.eval().to(self.device)

        return sampler.to(self.device)

    def load_mel(self):
        from playdiffusion.models.mel_spectrogram.mel import MelSpectrogram

        return MelSpectrogram.load_preset()

    def load_inpainter(self, config: dict):
        from playdiffusion.models.inpainter.masklm_text import load_maskgct_inpainter

        print(f"Using inpainter checkpoint: {config['checkpoint']}")
        return load_maskgct_inpainter(**config, device=self.device)
