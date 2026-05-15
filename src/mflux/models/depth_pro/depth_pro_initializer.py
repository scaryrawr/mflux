from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.depth_pro.model.depth_pro_model import DepthProModel
from mflux.models.depth_pro.weights.depth_pro_weight_definition import DepthProWeightDefinition


class DepthProInitializer:
    @staticmethod
    def init(
        model: DepthProModel,
        quantization: QuantizationConfig,
    ) -> None:
        # 1. Load weights using unified loader (handles download from Apple CDN)
        weights = WeightLoader.load(weight_definition=DepthProWeightDefinition)

        # 2. Apply weights and quantize using unified applier
        quantization = WeightApplier.apply_and_quantize(
            weights=weights,
            quantization=quantization,
            weight_definition=DepthProWeightDefinition,
            models={
                "depth_pro": model,
            },
        )
        WeightApplier.set_quantization_state(model, quantization)
