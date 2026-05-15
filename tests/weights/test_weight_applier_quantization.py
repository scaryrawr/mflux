from unittest.mock import Mock, patch

import pytest

from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition


class FakeWeightDefinition:
    quantization_predicate = staticmethod(lambda path, module: True)

    @staticmethod
    def get_components():
        return [ComponentDefinition(name="model", hf_subdir="")]


@pytest.mark.fast
def test_quantize_forwards_mode_and_group_size():
    model = Mock()
    quantization = QuantizationConfig(bits=4, mode="mxfp4", group_size=32)

    with patch("mflux.models.common.weights.loading.weight_applier.nn.quantize") as quantize:
        WeightApplier._quantize(
            models={"model": model},
            quantization=quantization,
            components={"model": ComponentDefinition(name="model", hf_subdir="")},
            weight_definition=FakeWeightDefinition,
        )

    quantize.assert_called_once_with(
        model,
        class_predicate=FakeWeightDefinition.quantization_predicate,
        bits=4,
        mode="mxfp4",
        group_size=32,
    )


@pytest.mark.fast
def test_set_quantization_state_exposes_compatibility_fields():
    model = Mock()
    quantization = QuantizationConfig(bits=8, mode="mxfp8", group_size=32)

    WeightApplier.set_quantization_state(model, quantization)

    assert model.quantization == quantization
    assert model.bits == 8
    assert model.q_mode == "mxfp8"
    assert model.q_group_size == 32


@pytest.mark.fast
def test_prequantized_weights_use_stored_mode_and_group_size():
    model = Mock()
    weight = object()
    weights = LoadedWeights(
        components={"model": {"weight": weight}},
        meta_data=MetaData(
            quantization_level=4,
            quantization_mode="mxfp4",
            quantization_group_size=32,
        ),
    )

    with patch("mflux.models.common.weights.loading.weight_applier.nn.quantize") as quantize:
        resolved = WeightApplier.apply_and_quantize(
            weights=weights,
            models={"model": model},
            quantization=QuantizationConfig(bits=4, mode="affine", group_size=64),
            weight_definition=FakeWeightDefinition,
        )

    assert resolved == QuantizationConfig(bits=4, mode="mxfp4", group_size=32)
    quantize.assert_called_once_with(
        model,
        class_predicate=FakeWeightDefinition.quantization_predicate,
        bits=4,
        mode="mxfp4",
        group_size=32,
    )
    model.update.assert_called_once_with({"weight": weight}, strict=False)


@pytest.mark.fast
def test_prequantized_single_component_uses_stored_mode_and_group_size():
    model = Mock()
    component = ComponentDefinition(name="model", hf_subdir="")
    weight = object()
    weights = LoadedWeights(
        components={"model": {"weight": weight}},
        meta_data=MetaData(
            quantization_level=8,
            quantization_mode="mxfp8",
            quantization_group_size=32,
        ),
    )

    with patch("mflux.models.common.weights.loading.weight_applier.nn.quantize") as quantize:
        resolved = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=model,
            component=component,
            quantization=QuantizationConfig(bits=8),
            quantization_predicate=FakeWeightDefinition.quantization_predicate,
        )

    assert resolved == QuantizationConfig(bits=8, mode="mxfp8", group_size=32)
    quantize.assert_called_once_with(
        model,
        class_predicate=FakeWeightDefinition.quantization_predicate,
        bits=8,
        mode="mxfp8",
        group_size=32,
    )
    model.update.assert_called_once_with({"weight": weight}, strict=False)
