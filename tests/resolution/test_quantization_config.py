import pytest

from mflux.models.common.resolution.quantization_config import QuantizationConfig


@pytest.mark.fast
def test_quantize_only_defaults_to_affine():
    config = QuantizationConfig.from_request(quantize=4)
    assert config == QuantizationConfig(bits=4, mode="affine", group_size=None)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("q_mode", "expected_bits"),
    [
        ("mxfp4", 4),
        ("nvfp4", 4),
        ("mxfp8", 8),
    ],
)
def test_q_mode_inferrable_bits(q_mode, expected_bits):
    config = QuantizationConfig.from_request(quantize=None, q_mode=q_mode)
    assert config == QuantizationConfig(bits=expected_bits, mode=q_mode, group_size=None)


@pytest.mark.fast
def test_q_mode_with_group_size_infers_bits():
    config = QuantizationConfig.from_request(quantize=None, q_mode="mxfp8", q_group_size=32)
    assert config == QuantizationConfig(bits=8, mode="mxfp8", group_size=32)


@pytest.mark.fast
def test_affine_q_mode_requires_quantize():
    with pytest.raises(ValueError, match="--quantize"):
        QuantizationConfig.from_request(quantize=None, q_mode="affine")


@pytest.mark.fast
def test_q_group_size_requires_quantize():
    with pytest.raises(ValueError, match="--quantize"):
        QuantizationConfig.from_request(quantize=None, q_group_size=32)


@pytest.mark.fast
def test_q_group_size_must_be_positive():
    with pytest.raises(ValueError, match="--q-group-size must be > 0"):
        QuantizationConfig.from_request(quantize=4, q_group_size=0)


@pytest.mark.fast
def test_mode_specific_bits_must_match():
    with pytest.raises(ValueError, match="mxfp4 requires --quantize 4"):
        QuantizationConfig.from_request(quantize=8, q_mode="mxfp4")


@pytest.mark.fast
def test_legacy_stored_quantization_defaults_to_affine_group_size():
    config = QuantizationConfig.from_stored(quantization_level=4)
    assert config == QuantizationConfig(bits=4, mode="affine", group_size=64)


@pytest.mark.fast
def test_model_metadata_uses_compatibility_keys():
    config = QuantizationConfig(bits=4, mode="mxfp4", group_size=32)

    metadata = config.to_model_metadata()

    assert metadata == {
        "quantization_level": "4",
        "quantization_mode": "mxfp4",
        "quantization_group_size": "32",
    }
