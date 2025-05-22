from typing import Tuple, Union

from fairseq2.models.wav2vec2._factory import (
     Wav2Vec2Factory,
     Wav2Vec2Config,
     Wav2Vec2EncoderConfig,
)
from fairseq2.models.wav2vec2._model import Wav2Vec2Model
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.typing import DataType, Device


def _encoder_xlsr2_1b_v2() -> Wav2Vec2EncoderConfig:
    """
    This is ported from the seamless_communication github repo
    Source:
    https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/models/unit_extractor/wav2vec2_layer_output.py
    """
    layer_descs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    return Wav2Vec2EncoderConfig(
        model_dim=1280,
        max_seq_len=4096,
        feature_dim=512,
        use_fbank=False,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=layer_descs,  # type: ignore
        feature_extractor_bias=True,
        feature_extractor_layer_norm_convs=True,
        feature_gradient_scale=1.0,
        num_fbank_channels=0,
        fbank_stride=0,
        sample_fbank_every_k=0,
        pos_encoder_type="conv",
        pos_encoder_depth=1,
        pos_conv_kernel_size=128,
        num_pos_conv_groups=16,
        use_conformer=False,
        num_encoder_layers=48,
        num_encoder_attn_heads=16,
        ffn_inner_dim=5120,
        dropout_p=0.1,
        attn_dropout_p=0.1,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.PRE,
        depthwise_conv_kernel_size=0,
    )


def _xlsr2_1b_v2() -> Wav2Vec2Config:
    """
    This is ported from the seamless_communication github repo
    Source:
    https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/models/unit_extractor/wav2vec2_layer_output.py
    """
    encoder_config = _encoder_xlsr2_1b_v2()

    return Wav2Vec2Config(
        encoder_config=encoder_config,
        final_dim=1024,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=1024,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
    )


def load_xlsr_encoder(
    device: Device, dtype: DataType, max_layer: Union[int, None] = 35
) -> Tuple[Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2EncoderConfig]:
    """
    build_xlsr_1b_v2

    Create the correct configs but
    Change the number of encoder (transformer) layers to
    avoid having extra weights loaded
    """
    encoder_config = _encoder_xlsr2_1b_v2()
    if max_layer is not None:
        encoder_config.num_encoder_layers = max_layer
    config = _xlsr2_1b_v2()
    config.encoder_config = encoder_config

    # Build the model
    model_builder = Wav2Vec2Factory(
        config=config
    )
    model = model_builder.create_model().to(device=device, dtype=dtype)
    return model, config, encoder_config
