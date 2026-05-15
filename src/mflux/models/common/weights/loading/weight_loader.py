import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.resolution.quantization_config import QuantizationConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.safetensors_reader import SafetensorsReader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType

logger = logging.getLogger(__name__)


class WeightLoader:
    @staticmethod
    def load_single(
        component: ComponentDefinition,
        repo_id: str,
        file_pattern: str = "*.safetensors",
    ) -> LoadedWeights:
        root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=[file_pattern, "config.json"]))
        weights, meta_data = WeightLoader._load_component(root_path, component)
        return LoadedWeights(
            components={component.name: weights},
            meta_data=meta_data,
        )

    @staticmethod
    def load(
        weight_definition: "WeightDefinitionType",
        model_path: str | None = None,
    ) -> LoadedWeights:
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=weight_definition.get_download_patterns(),
        )

        # 2. Load each component (with caching for shared sources)
        components = {}
        meta_data = MetaData()
        raw_weights_cache: dict[tuple, dict] = {}  # Cache by (path, loading_mode, weight_files)

        for component in weight_definition.get_components():
            weights, component_meta_data = WeightLoader._load_component(root_path, component, raw_weights_cache)
            components[component.name] = weights

            # Track metadata from first component that has it
            if not meta_data.has_mflux_metadata and component_meta_data.has_mflux_metadata:
                meta_data = component_meta_data

        return LoadedWeights(
            components=components,
            meta_data=meta_data,
        )

    @staticmethod
    def _load_component(
        root_path: Path | None,
        component: ComponentDefinition,
        raw_weights_cache: dict[tuple, dict] | None = None,
    ) -> tuple[dict, MetaData]:
        # Handle direct URL downloads (e.g., Apple CDN for DepthPro)
        if component.download_url is not None:
            file_path = WeightLoader._download_from_url(component.download_url, component.name)
            raw_weights = WeightLoader._load_weights_file(file_path, component.loading_mode)
        else:
            if root_path is None:
                raise ValueError(f"No root_path and no download_url for component: {component.name}")
            component_path = root_path / component.hf_subdir

            # Try mflux saved format first (including FP8 components reloaded after mflux-save).
            weights, meta_data = WeightLoader._try_load_mflux_format(component_path)
            if weights is not None:
                return weights, meta_data

            # Check cache for shared loading (e.g., FIBO VLM decoder + visual from same source)
            cache_key = (str(component_path), component.loading_mode, tuple(component.weight_files or []))
            if raw_weights_cache is not None and cache_key in raw_weights_cache:
                raw_weights = raw_weights_cache[cache_key]
            else:
                # Fall back to HuggingFace format with mapping
                raw_weights = WeightLoader._load_safetensors(
                    component_path, component.loading_mode, component.weight_files
                )
                # Cache for potential reuse by other components
                if raw_weights_cache is not None:
                    raw_weights_cache[cache_key] = raw_weights

        # Apply prefix filtering if specified (e.g., filter "model.language_model" vs "model.visual")
        if component.weight_prefix_filters is not None:
            raw_weights = {
                k: v
                for k, v in raw_weights.items()
                if any(k.startswith(prefix) for prefix in component.weight_prefix_filters)
            }

        if component.key_transform is not None:
            transformed_weights = {}
            for key, value in raw_weights.items():
                transformed_key = component.key_transform(key)
                if transformed_key is not None:
                    transformed_weights[transformed_key] = value
            raw_weights = transformed_weights

        if component.weight_transform is not None:
            raw_weights = {k: component.weight_transform(k, v) for k, v in raw_weights.items()}

        # Apply precision conversion if specified
        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        # Passthrough mode: apply bulk transform and unflatten (no key mapping)
        if component.mapping_getter is None:
            if component.bulk_transform is not None:
                raw_weights = {k: component.bulk_transform(v) for k, v in raw_weights.items()}
            return tree_unflatten(list(raw_weights.items())), MetaData()

        # Standard mode: apply declarative weight mapping
        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, MetaData()

    @staticmethod
    def _try_load_mflux_format(path: Path) -> tuple[dict | None, MetaData]:
        if not path.exists():
            return None, MetaData()

        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, MetaData()

        data = mx.load(str(shard_files[0]), return_metadata=True)
        metadata = WeightLoader._mflux_metadata(data[1] if len(data) > 1 else None)
        if metadata is None:
            metadata = WeightLoader._mflux_metadata(WeightLoader._index_metadata(path))
        if metadata is None:
            return None, MetaData()

        meta_data = WeightLoader._parse_mflux_metadata(metadata)

        # Load all shards
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            shard_data = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(shard_data[0].items()))

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, meta_data

    @staticmethod
    def _parse_mflux_metadata(metadata: dict) -> MetaData:
        quantization = QuantizationConfig.from_stored(
            quantization_level=WeightLoader._metadata_int(metadata.get("quantization_level")),
            quantization_mode=WeightLoader._metadata_str(metadata.get("quantization_mode")),
            quantization_group_size=WeightLoader._metadata_int(metadata.get("quantization_group_size")),
        )
        return MetaData.from_quantization(
            quantization=quantization,
            mflux_version=WeightLoader._metadata_str(metadata.get("mflux_version")),
        )

    @staticmethod
    def _mflux_metadata(metadata: dict | None) -> dict | None:
        if metadata is None:
            return None
        if "quantization_level" in metadata or metadata.get("mflux_version") is not None:
            return metadata
        return None

    @staticmethod
    def _index_metadata(path: Path) -> dict | None:
        index_path = path / "model.safetensors.index.json"
        if not index_path.exists():
            return None
        with open(index_path) as f:
            index = json.load(f)
        metadata = index.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return None

    @staticmethod
    def _metadata_int(value: int | str | None) -> int | None:
        if value in (None, "None", "null", ""):
            return None
        return int(value)

    @staticmethod
    def _metadata_str(value: str | None) -> str | None:
        if value in (None, "None", "null", ""):
            return None
        return value

    @staticmethod
    def _download_from_url(url: str, component_name: str) -> Path:
        cache_dir = MFLUX_CACHE_DIR / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = cache_dir / filename

        if not file_path.exists():
            logger.info(f"Downloading {component_name} weights from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info(f"Downloaded to {file_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error(f"Failed to download: {e}")
                logger.info(f"Please manually download from: {url}")
                raise FileNotFoundError(f"Model file not found at {file_path}") from e

        return file_path

    @staticmethod
    def _load_weights_file(file_path: Path, loading_mode: str) -> dict[str, mx.array]:
        if loading_mode == "torch_checkpoint":
            return WeightLoader._load_torch_checkpoint(file_path)
        elif loading_mode in ("mlx_native", "single"):
            data = mx.load(str(file_path), return_metadata=True)
            return dict(data[0].items())
        else:
            raise ValueError(f"Unsupported loading mode for single file: {loading_mode}")

    @staticmethod
    def _load_torch_checkpoint(file_path: Path) -> dict[str, mx.array]:
        pt_weights = torch.load(file_path, map_location="cpu", weights_only=False)
        return {k: mx.array(v.numpy()) for k, v in pt_weights.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _load_safetensors(path: Path, loading_mode: str, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path, weight_files)
        elif loading_mode == "torch_convert":
            return WeightLoader._load_torch_convert(path, weight_files)
        elif loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path)
        elif loading_mode == "torch_bfloat16":
            return WeightLoader._load_torch_bfloat16(path)
        elif loading_mode == "single":
            return WeightLoader._load_single(path)
        elif loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path)
        elif loading_mode == "fp8_safetensors":
            return WeightLoader._load_fp8_safetensors(path)
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_mlx_native(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_torch_convert(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name

            # Use mx.load which handles bfloat16 natively
            file_weights = mx.load(str(file_path))

            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        return all_weights

    @staticmethod
    def _load_torch_bfloat16(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        weight_files = sorted(set(index["weight_map"].values()))

        all_weights: dict[str, mx.array] = {}
        for wf in weight_files:
            file_path = path / wf
            data = torch_load_file(str(file_path))
            for k, v in data.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                np_arr = v.detach().cpu().numpy()
                all_weights[k] = mx.array(np_arr)

        return all_weights

    @staticmethod
    def _load_single(path: Path) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_multi_glob(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _load_fp8_safetensors(path: Path) -> dict[str, mx.array]:
        return SafetensorsReader.read_directory(path)

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        return {k: v if v.dtype == precision else v.astype(precision) for k, v in weights.items()}
