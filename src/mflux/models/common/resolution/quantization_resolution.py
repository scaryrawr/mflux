import logging

from mflux.models.common.resolution.actions import QuantizationAction, Rule
from mflux.models.common.resolution.quantization_config import QuantizationConfig

logger = logging.getLogger(__name__)


class QuantizationResolution:
    RULES = frozenset(
        {
            Rule(priority=0, name="none", check="none_none", action=QuantizationAction.NONE),
            Rule(priority=1, name="on_the_fly", check="none_any", action=QuantizationAction.REQUESTED),
            Rule(priority=2, name="pre_quantized", check="any_none", action=QuantizationAction.STORED),
            Rule(priority=3, name="compatible", check="any_compatible", action=QuantizationAction.STORED),
            Rule(priority=4, name="conflict", check="any_any", action=QuantizationAction.STORED),
        }
    )

    @staticmethod
    def resolve(
        stored: QuantizationConfig,
        requested: QuantizationConfig,
    ) -> tuple[QuantizationConfig, str | None]:
        for rule in sorted(QuantizationResolution.RULES, key=lambda r: r.priority):
            if QuantizationResolution._check(rule.check, stored, requested):
                QuantizationResolution._log(stored, requested, rule)
                return QuantizationResolution._execute(rule, stored, requested)

        raise ValueError(f"Unexpected quantization state: stored={stored.describe()}, requested={requested.describe()}")

    @staticmethod
    def _log(
        stored: QuantizationConfig,
        requested: QuantizationConfig,
        rule: Rule,
    ) -> None:
        logger.debug(
            f"Quantization resolution: stored={stored.describe()}, requested={requested.describe()} "
            f"→ rule '{rule.name}' ({rule.action.value})"
        )

    @staticmethod
    def _check(check: str, stored: QuantizationConfig, requested: QuantizationConfig) -> bool:
        if check == "none_none":
            return not stored.is_quantized and not requested.is_quantized
        if check == "none_any":
            return not stored.is_quantized and requested.is_quantized
        if check == "any_none":
            return stored.is_quantized and not requested.is_quantized
        if check == "any_compatible":
            return stored.is_quantized and requested.is_quantized and QuantizationResolution._is_compatible(stored, requested)
        if check == "any_any":
            return stored.is_quantized and requested.is_quantized
        return False

    @staticmethod
    def _execute(
        rule: Rule,
        stored: QuantizationConfig,
        requested: QuantizationConfig,
    ) -> tuple[QuantizationConfig, str | None]:
        if rule.action == QuantizationAction.NONE:
            return QuantizationConfig(), None
        if rule.action == QuantizationAction.REQUESTED:
            return requested, None
        if rule.action == QuantizationAction.STORED:
            warning = QuantizationResolution._conflict_warning(stored, requested) if rule.name == "conflict" else None
            return stored, warning
        return QuantizationConfig(), None

    @staticmethod
    def _is_compatible(stored: QuantizationConfig, requested: QuantizationConfig) -> bool:
        return (
            stored.bits == requested.bits
            and stored.mode == requested.mode
            and (requested.group_size is None or stored.group_size == requested.group_size)
        )

    @staticmethod
    def _conflict_warning(stored: QuantizationConfig, requested: QuantizationConfig) -> str:
        return f"Model is pre-quantized as {stored.describe()}. Ignoring requested {requested.describe()}."
