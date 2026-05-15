import pytest

from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.common.resolution.quantization_resolution import QuantizationResolution


class TestDecideQuantization:
    @pytest.mark.fast
    def test_no_quantization_when_neither_specified(self):
        quantization, warning = QuantizationResolution.resolve(stored=QuantizationConfig(), requested=QuantizationConfig())
        assert quantization.bits is None
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize("requested_bits", [3, 4, 5, 6, 8])
    def test_on_the_fly_quantization(self, requested_bits):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(),
            requested=QuantizationConfig(bits=requested_bits),
        )
        assert quantization.bits == requested_bits
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize("stored_bits", [3, 4, 5, 6, 8])
    def test_prequantized_no_request(self, stored_bits):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=stored_bits),
            requested=QuantizationConfig(),
        )
        assert quantization.bits == stored_bits
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize("bits_value", [3, 4, 5, 6, 8])
    def test_prequantized_matching_request(self, bits_value):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=bits_value),
            requested=QuantizationConfig(bits=bits_value),
        )
        assert quantization.bits == bits_value
        assert warning is None

    @pytest.mark.fast
    def test_prequantized_matching_request_with_unspecified_group_size(self):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=4, mode="affine", group_size=64),
            requested=QuantizationConfig(bits=4, mode="affine", group_size=None),
        )
        assert quantization == QuantizationConfig(bits=4, mode="affine", group_size=64)
        assert warning is None

    @pytest.mark.fast
    def test_prequantized_matching_mode_request(self):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=4, mode="mxfp4"),
            requested=QuantizationConfig(bits=4, mode="mxfp4"),
        )
        assert quantization == QuantizationConfig(bits=4, mode="mxfp4")
        assert warning is None

    @pytest.mark.fast
    def test_on_the_fly_quantization_preserves_mode_and_group_size(self):
        requested = QuantizationConfig(bits=8, mode="mxfp8", group_size=32)

        quantization, warning = QuantizationResolution.resolve(stored=QuantizationConfig(), requested=requested)

        assert quantization == requested
        assert warning is None

    @pytest.mark.fast
    def test_prequantized_explicit_group_size_mismatch_warns(self):
        stored = QuantizationConfig(bits=4, mode="affine", group_size=64)
        requested = QuantizationConfig(bits=4, mode="affine", group_size=32)

        quantization, warning = QuantizationResolution.resolve(stored=stored, requested=requested)

        assert quantization == stored
        assert warning is not None
        assert stored.describe() in warning
        assert requested.describe() in warning

    @pytest.mark.fast
    def test_prequantized_missing_group_size_mismatches_explicit_request(self):
        stored = QuantizationConfig(bits=8, mode="mxfp8")
        requested = QuantizationConfig(bits=8, mode="mxfp8", group_size=32)

        quantization, warning = QuantizationResolution.resolve(stored=stored, requested=requested)

        assert quantization == stored
        assert warning is not None
        assert stored.describe() in warning
        assert requested.describe() in warning

    @pytest.mark.fast
    def test_prequantized_mode_mismatch_warns(self):
        stored = QuantizationConfig(bits=4, mode="mxfp4")
        requested = QuantizationConfig(bits=4, mode="nvfp4")

        quantization, warning = QuantizationResolution.resolve(stored=stored, requested=requested)

        assert quantization == stored
        assert warning is not None
        assert stored.describe() in warning
        assert requested.describe() in warning

    @pytest.mark.fast
    def test_prequantized_explicit_group_size_mismatch_uses_stored(self):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=4, mode="affine", group_size=64),
            requested=QuantizationConfig(bits=4, mode="affine", group_size=32),
        )
        assert quantization == QuantizationConfig(bits=4, mode="affine", group_size=64)
        assert warning is not None

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "stored_bits,requested_bits",
        [(4, 8), (8, 4), (4, 3), (3, 8), (6, 4)],
    )
    def test_prequantized_conflicting_request_uses_stored(self, stored_bits, requested_bits):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=stored_bits),
            requested=QuantizationConfig(bits=requested_bits),
        )
        assert quantization.bits == stored_bits

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "stored_bits,requested_bits",
        [(4, 8), (8, 4), (4, 3)],
    )
    def test_prequantized_conflicting_request_warns(self, stored_bits, requested_bits):
        quantization, warning = QuantizationResolution.resolve(
            stored=QuantizationConfig(bits=stored_bits),
            requested=QuantizationConfig(bits=requested_bits),
        )
        assert warning is not None
        assert QuantizationConfig(bits=stored_bits).describe() in warning
        assert QuantizationConfig(bits=requested_bits).describe() in warning


class TestQuantizationPolicyCompleteness:
    VALID_QUANT_LEVELS = [None, 3, 4, 5, 6, 8]

    @pytest.mark.fast
    def test_all_combinations_handled(self):
        for stored in self.VALID_QUANT_LEVELS:
            for requested in self.VALID_QUANT_LEVELS:
                quantization, warning = QuantizationResolution.resolve(
                    stored=QuantizationConfig(bits=stored),
                    requested=QuantizationConfig(bits=requested),
                )
                assert quantization.bits is None or quantization.bits in self.VALID_QUANT_LEVELS

    @pytest.mark.fast
    def test_result_is_always_stored_or_requested_or_none(self):
        for stored in self.VALID_QUANT_LEVELS:
            for requested in self.VALID_QUANT_LEVELS:
                quantization, _ = QuantizationResolution.resolve(
                    stored=QuantizationConfig(bits=stored),
                    requested=QuantizationConfig(bits=requested),
                )
                assert quantization.bits in (stored, requested, None)
