from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType


class WeightApplier:
    @staticmethod
    def set_quantization_state(model, quantization: QuantizationConfig) -> None:
        model.quantization = quantization
        model.bits = quantization.bits
        model.q_mode = quantization.metadata_mode
        model.q_group_size = quantization.metadata_group_size

    @staticmethod
    def apply_and_quantize_single(
        weights: LoadedWeights,
        model: nn.Module,
        component: ComponentDefinition,
        quantization: QuantizationConfig,
        quantization_predicate=None,
    ) -> QuantizationConfig:
        stored_q = QuantizationConfig.from_stored(
            quantization_level=weights.meta_data.quantization_level,
            quantization_mode=weights.meta_data.quantization_mode,
            quantization_group_size=weights.meta_data.quantization_group_size,
        )
        component_weights = weights.components.get(component.name)

        if component_weights is None:
            raise ValueError(f"No weights found for component: {component.name}")

        if quantization_predicate is None:

            def quantization_predicate(path, module):
                return hasattr(module, "to_quantized")

        quantization, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantization)

        if warning:
            print(f"⚠️  {warning}")

        if not quantization.is_quantized:
            model.update(component_weights, strict=False)
        elif not stored_q.is_quantized:
            model.update(component_weights, strict=False)
            if not component.skip_quantization:
                nn.quantize(
                    model,
                    class_predicate=quantization_predicate,
                    bits=quantization.bits,
                    mode=quantization.mode,
                    group_size=quantization.group_size,
                )
        else:
            if not component.skip_quantization:
                nn.quantize(
                    model,
                    class_predicate=quantization_predicate,
                    bits=quantization.bits,
                    mode=quantization.mode,
                    group_size=quantization.group_size,
                )
            model.update(component_weights, strict=False)

        return quantization

    @staticmethod
    def apply_and_quantize(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        quantization: QuantizationConfig,
        weight_definition: "WeightDefinitionType",
    ) -> QuantizationConfig:
        stored_q = QuantizationConfig.from_stored(
            quantization_level=weights.meta_data.quantization_level,
            quantization_mode=weights.meta_data.quantization_mode,
            quantization_group_size=weights.meta_data.quantization_group_size,
        )
        components = {c.name: c for c in weight_definition.get_components()}

        quantization, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantization)

        if warning:
            print(f"⚠️  {warning}")

        if not quantization.is_quantized:
            WeightApplier._set_weights(weights, models, components)
        elif not stored_q.is_quantized:
            WeightApplier._set_weights(weights, models, components)
            WeightApplier._quantize(models, quantization, components, weight_definition)
        else:
            WeightApplier._quantize(models, quantization, components, weight_definition)
            WeightApplier._set_weights(weights, models, components)

        return quantization

    @staticmethod
    def _set_weights(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        components: dict | None = None,
    ) -> None:
        for name, model in models.items():
            component_weights = weights.components.get(name)
            if component_weights is not None:
                if components is not None:
                    component = components.get(name)
                    if component is not None and component.weight_subkey is not None:
                        component_weights = component_weights.get(component.weight_subkey, component_weights)
                model.update(component_weights, strict=False)

    @staticmethod
    def _quantize(
        models: dict[str, nn.Module],
        quantization: QuantizationConfig,
        components: dict,
        weight_definition: "WeightDefinitionType",
    ) -> None:
        for name, model in models.items():
            component = components.get(name)
            if component and component.skip_quantization:
                continue
            nn.quantize(
                model,
                class_predicate=weight_definition.quantization_predicate,
                bits=quantization.bits,
                mode=quantization.mode,
                group_size=quantization.group_size,
            )
