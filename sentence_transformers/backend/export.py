from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import transformers
from huggingface_hub import hf_hub_download, list_repo_files
from packaging import version
from torch import Tensor, nn

from sentence_transformers.backend.utils import backend_should_export

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

logger = logging.getLogger(__name__)

EXPORT_CONFIG_FILE = "export.json"
MODEL_FILE_NAMES = {"onnx": "model.onnx", "openvino": "openvino_model.xml"}
MODEL_FILE_GLOBS = {"onnx": "*.onnx", "openvino": "openvino*.xml"}
BACKEND_NAMES = {"onnx": "ONNX", "openvino": "OpenVINO"}
# first version to include `XxxExporter`, `XxxModelRunner`, XxxQuantizer` is v5.20
MIN_TRANSFORMERS_VERSION = "5.19.0.dev0"


def is_transformers_export_available(backend: str | None = None) -> tuple[bool, str]:
    installed = version.parse(transformers.__version__)
    if installed.release < version.parse(MIN_TRANSFORMERS_VERSION).release:
        target = BACKEND_NAMES.get(backend, "ONNX or OpenVINO")
        return False, (
            f"exporting the whole model to {target} requires `transformers>={MIN_TRANSFORMERS_VERSION}` "
            f"(found {transformers.__version__})"
        )
    return True, ""


class _ExportWrapper(nn.Module):
    """Runs the modules of a model on keyword features and returns the requested output features."""

    def __init__(self, model: BaseModel, output_names: list[str]):
        super().__init__()
        self.model = model
        self.output_names = output_names

    def forward(self, **features: Tensor) -> dict[str, Tensor]:
        outputs = self.model(dict(features))
        return {name: outputs[name] for name in self.output_names}


def export_model(
    model: BaseModel,
    backend: Literal["onnx", "openvino"],
    output_directory: str | Path,
    quantizer: Any = None,
) -> Path:
    available, reason = is_transformers_export_available(backend)
    if not available:
        raise ImportError(f"Cannot export the whole model to {BACKEND_NAMES[backend]}: {reason}.")

    if model.modalities != ["text"]:
        raise ValueError(
            f"Exporting the whole model to {BACKEND_NAMES[backend]} is only supported for text models, "
            f"but the model supports the modalities {model.modalities}."
        )

    output_directory = Path(output_directory)
    features = model.preprocess(model._export_sample_inputs())
    sample_inputs = {key: value for key, value in features.items() if isinstance(value, Tensor)}

    with torch.no_grad():
        outputs = model(dict(sample_inputs))
    output_names = [
        name
        for name, value in outputs.items()
        if isinstance(value, Tensor) and not any(value is sample for sample in sample_inputs.values())
    ]  # Outputs that are an input as it came are returned by the features of the model already
    wrapper = _ExportWrapper(model, output_names).eval()

    if backend == "onnx":
        from transformers.exporters import OnnxConfig, OnnxExporter

        exporter = OnnxExporter().export(wrapper, sample_inputs, config=OnnxConfig(dynamic=True, quantizer=quantizer))
    else:
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        exporter = OpenVINOExporter().export(
            wrapper, sample_inputs, config=OpenVINOConfig(dynamic=True, stateful=False, quantizer=quantizer)
        )
    exporter.save_pretrained(output_directory)
    if backend == "openvino":
        _rename_openvino_files(output_directory)

    return output_directory


def _rename_openvino_files(directory: Path) -> None:
    """Name the OpenVINO files `openvino_model.xml` and `openvino_model.bin` and update the config to match."""
    config_path = directory / EXPORT_CONFIG_FILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    component = config["components"]["model"]
    source = directory / component["file"]
    target = source.with_name(MODEL_FILE_NAMES["openvino"])
    if source != target:
        source.rename(target)
        source.with_suffix(".bin").rename(target.with_suffix(".bin"))
        component["file"] = target.name
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def new_export_directory() -> tempfile.TemporaryDirectory:
    """A temporary directory to export to, removed when the returned object is garbage collected."""
    return tempfile.TemporaryDirectory(prefix="sentence_transformers_export_")


BACKEND_MODEL_KWARGS = (
    "provider",
    "provider_options",
    "session_options",
    "ov_config",
    "file_name",
    "export",
    "subfolder",
)


def _raise_cannot_export(model_name_or_path: str, backend_name: str) -> None:
    raise ValueError(
        f"No {backend_name} model was found in {model_name_or_path!r} to load with Optimum, and a Transformer model "
        f"cannot be exported to {backend_name} on its own. Load the whole model with `backend={backend_name.lower()!r}` "
        "(e.g. `SentenceTransformer(...)`) to export it, which requires a version of `transformers` that ships "
        "the exporters and their runners."
    )


def _export_candidates(model_kwargs: dict[str, Any], backend: str) -> list[Path]:
    """The directories that may hold an export of the whole model, by order of preference."""
    subfolder = model_kwargs.get("subfolder") or ""
    file_name = model_kwargs.get("file_name")
    if file_name is not None and len(Path(file_name).parts) > 1:
        return [Path(subfolder, *Path(file_name).parts[:-1])]
    return [Path(subfolder), Path(subfolder, backend)]


def find_existing_export(
    model_name_or_path: str,
    backend: str,
    model_kwargs: dict[str, Any],
    token: bool | str | None = None,
    revision: Any = None,
) -> tuple[str, str | None] | None:
    """Look for an export of the whole model in ``model_name_or_path``, local or on the Hub.

    Returns:
        The ``(subfolder, file_name)`` of the export, with ``file_name`` None for the default graph,
        or None if there is no export of the whole model.
    """
    candidates = _export_candidates(model_kwargs, backend)
    load_path = Path(model_name_or_path)
    if load_path.exists():
        subfolder = next((c for c in candidates if (load_path / c / EXPORT_CONFIG_FILE).exists()), None)
    else:
        repo_files = set(
            list_repo_files(
                load_path.as_posix(),
                repo_type="model",
                revision=getattr(revision, "resolved", revision),
                token=token,
            )
        )
        subfolder = next((c for c in candidates if (c / EXPORT_CONFIG_FILE).as_posix() in repo_files), None)
    if subfolder is None:
        return None
    file_name = model_kwargs.get("file_name")
    return subfolder.as_posix(), Path(file_name).name if file_name is not None else None


def is_legacy_export(
    model_name_or_path: str,
    backend: str,
    model_kwargs: dict[str, Any],
    token: bool | str | None = None,
    revision: Any = None,
) -> bool:
    """Whether the files to load were exported by a previous version of sentence-transformers, which only exported the Transformer model."""
    load_path = Path(model_name_or_path)
    legacy_kwargs = {key: model_kwargs[key] for key in ("file_name", "subfolder") if key in model_kwargs}
    legacy_kwargs.update(token=token, revision=revision)
    should_export, _ = backend_should_export(
        load_path,
        load_path.exists(),
        legacy_kwargs,
        MODEL_FILE_NAMES[backend],
        MODEL_FILE_GLOBS[backend],
        BACKEND_NAMES[backend],
    )
    return not should_export


def legacy_export_message(model_name_or_path: str, backend: str) -> str:
    """The warning for loading files exported by a previous version of sentence-transformers, which only exported the Transformer model."""
    available, reason = is_transformers_export_available(backend)
    advice = (
        'To export the whole model instead, load the model with `model_kwargs={"export": True}` '
        "and save it with `save_pretrained()`."
        if available
        else f"To export the whole model instead, {reason}."
    )
    return (
        f"The {BACKEND_NAMES[backend]} model in {model_name_or_path!r} was exported with a previous version of "
        f"sentence-transformers, so it only contains the Transformer model: the other modules (e.g. pooling and normalization) run in PyTorch. {advice}"
    )


def check_can_export(model_name_or_path: str, backend: str) -> None:
    """Raise an ``ImportError`` if the whole model cannot be exported with ``backend``."""
    available, reason = is_transformers_export_available(backend)
    if not available:
        raise ImportError(
            f"No {BACKEND_NAMES[backend]} model was found in {model_name_or_path!r} to load, "
            f"and it cannot be exported: {reason}."
        )


def split_backend_kwargs(
    model_kwargs: dict[str, Any] | None, backend: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``model_kwargs`` into the ones for the backend runtime and the ones for ``from_pretrained``."""
    model_kwargs = dict(model_kwargs or {})
    backend_kwargs = {key: model_kwargs.pop(key) for key in BACKEND_MODEL_KWARGS if key in model_kwargs}
    if backend == "openvino" and "device" in model_kwargs:
        backend_kwargs["device"] = model_kwargs.pop("device")
    return backend_kwargs, model_kwargs


def read_export_config(directory: str | Path) -> dict[str, Any]:
    """The ``export.json`` of a saved export."""
    return json.loads((Path(directory) / EXPORT_CONFIG_FILE).read_text(encoding="utf-8"))


def export_graph_name(directory: str | Path) -> str:
    """The file name of the graph of a saved export, e.g. ``model.onnx``."""
    return read_export_config(directory)["components"]["model"]["file"]


def export_files(directory: str | Path) -> list[Path]:
    """The files making up a saved export: the graph, its weights, ``export.json`` and ``config.json``."""
    directory = Path(directory)
    graph = Path(export_graph_name(directory))
    # The weights are stored next to the graph: `model.bin`, or `model.onnx_data` / `model.onnx.data`
    weights = (
        [graph.with_suffix(".bin").name] if graph.suffix == ".xml" else [f"{graph.stem}.onnx_data", f"{graph}.data"]
    )
    names = [graph.name, *weights, EXPORT_CONFIG_FILE, "config.json"]
    return [directory / name for name in names if (directory / name).exists()]


def save_export(source_directory: str | Path, directory: str | Path) -> None:
    """Copy the files of a saved export into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for source in export_files(source_directory):
        target = directory / source.name
        if not (target.exists() and target.samefile(source)):
            shutil.copy(source, target)


def resolve_export_directory(
    model_name_or_path: str,
    subfolder: str | None = None,
    token: bool | str | None = None,
    revision: Any = None,
    cache_dir: str | None = None,
) -> Path:
    """The local directory of an export, in a local directory or a cached snapshot of a Hub repository."""
    if Path(model_name_or_path).is_dir():
        return Path(model_name_or_path, subfolder or "")
    config = hf_hub_download(
        model_name_or_path,
        (Path(subfolder or "") / EXPORT_CONFIG_FILE).as_posix(),
        revision=getattr(revision, "resolved", revision),
        token=token,
        cache_dir=cache_dir,
    )
    return Path(config).parent
