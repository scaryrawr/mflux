import os
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.variants import ZImageTurbo
from mflux.utils.version_util import VersionUtil

PATH = "tests/4bit/"


@pytest.mark.fast
def test_legacy_model_metadata_normalizes_to_affine_group_size():
    meta_data = WeightLoader._parse_mflux_metadata({"quantization_level": "4"})

    assert meta_data.quantization == QuantizationConfig(bits=4, mode="affine", group_size=64)
    assert meta_data.quantization_level == 4
    assert meta_data.quantization_mode == "affine"
    assert meta_data.quantization_group_size == 64


@pytest.mark.fast
def test_model_metadata_preserves_mode_and_group_size():
    meta_data = WeightLoader._parse_mflux_metadata(
        {
            "quantization_level": "4",
            "quantization_mode": "mxfp4",
            "quantization_group_size": "32",
            "mflux_version": "test-version",
        }
    )

    assert meta_data.quantization == QuantizationConfig(bits=4, mode="mxfp4", group_size=32)
    assert meta_data.mflux_version == "test-version"


@pytest.mark.fast
def test_unquantized_model_metadata_preserves_version():
    meta_data = WeightLoader._parse_mflux_metadata(
        {
            "quantization_level": "None",
            "quantization_mode": "None",
            "quantization_group_size": "None",
            "mflux_version": "test-version",
        }
    )

    assert not meta_data.quantization.is_quantized
    assert meta_data.has_mflux_metadata
    assert meta_data.mflux_version == "test-version"


@pytest.mark.fast
def test_model_saver_metadata_uses_quantization_config():
    metadata = ModelSaver._metadata(QuantizationConfig(bits=4, mode="mxfp4", group_size=32))

    assert metadata["quantization_level"] == "4"
    assert metadata["quantization_mode"] == "mxfp4"
    assert metadata["quantization_group_size"] == "32"
    assert metadata["mflux_version"] == VersionUtil.get_mflux_version()


@pytest.mark.fast
def test_model_saver_round_trip_preserves_quantization_metadata(tmp_path):
    model = nn.Linear(2, 2)
    quantization = QuantizationConfig(bits=8, mode="mxfp8", group_size=32)

    ModelSaver._save_weights(str(tmp_path), quantization, model, "tiny")
    loaded_weights, meta_data = WeightLoader._try_load_mflux_format(tmp_path / "tiny")

    assert loaded_weights is not None
    assert meta_data.quantization == quantization
    assert meta_data.quantization_level == 8
    assert meta_data.quantization_mode == "mxfp8"
    assert meta_data.quantization_group_size == 32
    mx.eval(loaded_weights["weight"], loaded_weights["bias"])


class TestModelSaving:
    @pytest.mark.slow
    def test_save_and_load_4bit_model(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSaving.delete_folder_if_exists(PATH)

        try:
            # given a saved quantized model (and an image from that model)
            modelA = ZImageTurbo(quantize=4)
            image1 = modelA.generate_image(
                seed=42,
                prompt="Luxury food photograph",
                num_inference_steps=2,
                height=368,
                width=640,
            )
            modelA.save_model(PATH)
            del modelA

            # Verify that the mflux version is correctly saved in the model's metadata
            _, meta_data = WeightLoader._try_load_mflux_format(Path(PATH) / "vae")
            assert meta_data.mflux_version == VersionUtil.get_mflux_version(), "mflux version not correctly saved in metadata"  # fmt: off
            assert meta_data.quantization_level == 4, "quantization level not correctly saved in metadata"  # fmt: off
            assert meta_data.quantization_mode == "affine", "quantization mode not correctly saved in metadata"  # fmt: off

            # when loading the quantized model (also without specifying bits)
            modelB = ZImageTurbo(model_path=PATH)

            # then we can load the model and generate the identical image
            image2 = modelB.generate_image(
                seed=42,
                prompt="Luxury food photograph",
                num_inference_steps=2,
                height=368,
                width=640,
            )
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1.",
            )

        finally:
            # cleanup
            TestModelSaving.delete_folder(PATH)

    @staticmethod
    def delete_folder(path: str) -> None:
        return shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print(f"Folder does not exist: {path}")
