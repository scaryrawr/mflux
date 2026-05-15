from dataclasses import dataclass

QUANTIZATION_MODE_AFFINE = "affine"
QUANTIZATION_MODE_CHOICES = [QUANTIZATION_MODE_AFFINE, "mxfp4", "nvfp4", "mxfp8"]
QUANTIZATION_MODE_BITS = {
    "mxfp4": 4,
    "nvfp4": 4,
    "mxfp8": 8,
}
LEGACY_AFFINE_GROUP_SIZE = 64


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int | None = None
    mode: str = QUANTIZATION_MODE_AFFINE
    group_size: int | None = None

    @property
    def is_quantized(self) -> bool:
        return self.bits is not None

    @property
    def metadata_mode(self) -> str | None:
        return self.mode if self.is_quantized else None

    @property
    def metadata_group_size(self) -> int | None:
        return self.group_size if self.is_quantized else None

    @classmethod
    def from_request(
        cls,
        quantize: int | None,
        q_mode: str | None = None,
        q_group_size: int | None = None,
    ) -> "QuantizationConfig":
        mode = q_mode or QUANTIZATION_MODE_AFFINE
        inferred_bits = QUANTIZATION_MODE_BITS.get(mode)
        bits = quantize if quantize is not None else inferred_bits
        require_bits = q_group_size is not None or (q_mode is not None and inferred_bits is None)

        config = cls(bits=bits, mode=mode, group_size=q_group_size)
        config.validate(require_bits=require_bits)
        return config

    @classmethod
    def from_stored(
        cls,
        quantization_level: int | None,
        quantization_mode: str | None = None,
        quantization_group_size: int | None = None,
    ) -> "QuantizationConfig":
        mode = quantization_mode or QUANTIZATION_MODE_AFFINE
        group_size = quantization_group_size
        if quantization_level is not None and quantization_mode is None and quantization_group_size is None:
            group_size = LEGACY_AFFINE_GROUP_SIZE
        config = cls(bits=quantization_level, mode=mode, group_size=group_size)
        config.validate(require_bits=False)
        return config

    def validate(self, require_bits: bool) -> None:
        if self.mode not in QUANTIZATION_MODE_CHOICES:
            raise ValueError(f"Unsupported quantization mode: {self.mode}")
        if require_bits and self.bits is None:
            raise ValueError("--quantize must be provided when using --q-mode or --q-group-size")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("--q-group-size must be > 0")
        expected_bits = QUANTIZATION_MODE_BITS.get(self.mode)
        if self.bits is not None and expected_bits is not None and self.bits != expected_bits:
            raise ValueError(f"--q-mode {self.mode} requires --quantize {expected_bits}")

    def describe(self) -> str:
        if self.bits is None:
            return "unquantized"
        details = f"{self.bits}-bit {self.mode}"
        if self.group_size is not None:
            details += f" group-size {self.group_size}"
        return details

    def to_model_metadata(self) -> dict[str, str]:
        return {
            "quantization_level": str(self.bits),
            "quantization_mode": str(self.metadata_mode),
            "quantization_group_size": str(self.metadata_group_size),
        }
