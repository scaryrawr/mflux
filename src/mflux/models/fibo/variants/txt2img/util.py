import gc
import json

import mlx.core as mx

from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.utils.prompt_util import PromptUtil


class FiboUtil:
    @staticmethod
    def get_json_prompt(
        args,
        quantization: QuantizationConfig,
    ):
        prompt = PromptUtil.read_prompt(args)

        try:
            json.loads(prompt)
            json_prompt = prompt
        except json.JSONDecodeError:
            vlm = FiboVLM(quantization=quantization)
            json_prompt = vlm.generate(prompt=prompt, seed=42)
            del vlm
            gc.collect()
            mx.clear_cache()
        return json_prompt
