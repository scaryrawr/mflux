import inspect
import sys

import pytest

from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.variants.concept_attention.flux_concept import Flux1Concept
from mflux.models.flux.variants.concept_attention.flux_concept_from_image import Flux1ConceptFromImage
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev
from mflux.models.flux.variants.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.z_image.variants import ZImage, ZImageTurbo
from mflux.models.z_image.z_image_initializer import ZImageInitializer


@pytest.mark.fast
@pytest.mark.parametrize(
    "model_class",
    [
        DepthPro,
        ErnieImage,
        FIBO,
        FIBOEdit,
        FiboVLM,
        Flux1,
        Flux1Concept,
        Flux1ConceptFromImage,
        Flux1Controlnet,
        Flux1Depth,
        Flux1Fill,
        Flux1InContextDev,
        Flux1InContextFill,
        Flux1Kontext,
        Flux1Redux,
        Flux2Klein,
        Flux2KleinEdit,
        Ideogram4,
        QwenImage,
        QwenImageEdit,
        SeedVR2,
        ZImage,
        ZImageTurbo,
    ],
)
def test_public_model_constructor_accepts_quantization_mode_kwargs(model_class):
    parameters = inspect.signature(model_class).parameters

    assert parameters["q_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["q_group_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["quantization"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.fast
def test_z_image_turbo_forwards_quantization_mode_kwargs(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured["quantization"] = kwargs["quantization"]

    monkeypatch.setattr(ZImageInitializer, "init", staticmethod(fake_init))

    ZImageTurbo(q_mode="mxfp4", q_group_size=32)

    assert captured["quantization"] == QuantizationConfig(bits=4, mode="mxfp4", group_size=32)


@pytest.mark.fast
def test_ernie_image_forwards_quantization_mode_kwargs(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured["quantization"] = kwargs["quantization"]

    monkeypatch.setattr(ErnieImageInitializer, "init", staticmethod(fake_init))

    ErnieImage(q_mode="mxfp8", q_group_size=32)

    assert captured["quantization"] == QuantizationConfig(bits=8, mode="mxfp8", group_size=32)


@pytest.mark.fast
def test_ideogram4_forwards_quantization_mode_kwargs(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured["quantization"] = kwargs["quantization"]

    monkeypatch.setattr(Ideogram4Initializer, "init", staticmethod(fake_init))

    Ideogram4(q_mode="mxfp8", q_group_size=32)

    assert captured["quantization"] == QuantizationConfig(bits=8, mode="mxfp8", group_size=32)


@pytest.mark.fast
def test_save_cli_forwards_ernie_quantization_config(monkeypatch, tmp_path):
    from mflux.models.common.cli import save

    captured = {}

    class FakeErnieImage:
        def __init__(self, **kwargs) -> None:
            captured["model_kwargs"] = kwargs

        def save_model(self, path: str) -> None:
            captured["save_path"] = path

    monkeypatch.setattr(save, "ErnieImage", FakeErnieImage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-save",
            "--model",
            "baidu/ERNIE-Image-Turbo",
            "--q-mode",
            "mxfp8",
            "--q-group-size",
            "32",
            "--path",
            str(tmp_path / "ernie-mxfp8"),
        ],
    )

    save.main()

    assert captured["model_kwargs"]["quantization"] == QuantizationConfig(bits=8, mode="mxfp8", group_size=32)
    assert "quantize" not in captured["model_kwargs"]
    assert "q_mode" not in captured["model_kwargs"]
    assert "q_group_size" not in captured["model_kwargs"]
    assert captured["save_path"] == str(tmp_path / "ernie-mxfp8")


@pytest.mark.fast
def test_flux_from_name_forwards_quantization_mode_kwargs(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured["quantization"] = kwargs["quantization"]

    monkeypatch.setattr(FluxInitializer, "init", staticmethod(fake_init))

    Flux1.from_name("schnell", q_mode="mxfp8")

    assert captured["quantization"] == QuantizationConfig(bits=8, mode="mxfp8", group_size=None)
