from dataclasses import dataclass

from mflux.models.common.resolution.quantization_config import QuantizationConfig


@dataclass
class MetaData:
    quantization_level: int | None = None
    quantization_mode: str | None = None
    quantization_group_size: int | None = None
    mflux_version: str | None = None

    @property
    def has_mflux_metadata(self) -> bool:
        return self.quantization_level is not None or self.mflux_version is not None

    @property
    def quantization(self) -> QuantizationConfig:
        return QuantizationConfig.from_stored(
            quantization_level=self.quantization_level,
            quantization_mode=self.quantization_mode,
            quantization_group_size=self.quantization_group_size,
        )

    @classmethod
    def from_quantization(cls, quantization: QuantizationConfig, mflux_version: str | None = None) -> "MetaData":
        return cls(
            quantization_level=quantization.bits,
            quantization_mode=quantization.metadata_mode,
            quantization_group_size=quantization.metadata_group_size,
            mflux_version=mflux_version,
        )


@dataclass
class LoadedWeights:
    components: dict[str, dict]
    meta_data: MetaData

    def __getattr__(self, name: str) -> dict | None:
        if name in ("components", "meta_data"):
            return object.__getattribute__(self, name)
        return self.components.get(name)

    def num_transformer_blocks(self, component_name: str = "transformer") -> int:
        transformer = self.components.get(component_name)
        if transformer is None:
            for comp in self.components.values():
                if isinstance(comp, dict) and "transformer_blocks" in comp:
                    transformer = comp
                    break
        if transformer and "transformer_blocks" in transformer:
            return len(transformer["transformer_blocks"])
        return 0

    def num_single_transformer_blocks(self, component_name: str = "transformer") -> int:
        transformer = self.components.get(component_name)
        if transformer is None:
            for comp in self.components.values():
                if isinstance(comp, dict) and "single_transformer_blocks" in comp:
                    transformer = comp
                    break
        if transformer and "single_transformer_blocks" in transformer:
            return len(transformer["single_transformer_blocks"])
        return 0
