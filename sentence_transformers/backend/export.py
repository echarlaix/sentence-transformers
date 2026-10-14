from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
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
        # Saved with the export, to load it again
        self.config = model.config

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

        exporter = OnnxExporter().export(
            wrapper, sample_inputs, config=OnnxConfig(dynamic=True, quantizer=quantizer)
        )
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


@dataclass(frozen=True)
class BackendPlan:
    """How a model is loaded with a backend.

    Build one with :meth:`torch`, :meth:`legacy`, :meth:`export_full` or :meth:`load_full`.

    - ``"torch"``: no backend, the modules run in PyTorch.
    - ``"legacy"``: files exported with Optimum, which only hold the Transformer model and are loaded by its module.
    - ``"full"``: the whole model runs in the backend, from an export that is made (``export``) or already exists
      (in ``subfolder``, with ``file_name`` as the graph if it isn't the default one).

    ``message`` is a warning for the caller to log, and doesn't take part in comparisons.
    """

    mode: Literal["torch", "full", "legacy"]
    export: bool = False
    file_name: str | None = None
    subfolder: str | None = None
    message: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in ("torch", "full", "legacy"):
            raise ValueError(f"Unknown backend plan mode {self.mode!r}.")
        if self.mode != "full" and (self.export or self.file_name is not None or self.subfolder is not None):
            raise ValueError(f"A {self.mode!r} backend plan cannot export or point to an export of the whole model.")
        if self.export and (self.file_name is not None or self.subfolder is not None):
            raise ValueError("A backend plan that exports the whole model cannot point to an existing export.")

    @classmethod
    def torch(cls) -> BackendPlan:
        return cls("torch")

    @classmethod
    def legacy(cls, message: str | None = None) -> BackendPlan:
        return cls("legacy", message=message)

    @classmethod
    def export_full(cls, message: str | None = None) -> BackendPlan:
        return cls("full", export=True, message=message)

    @classmethod
    def load_full(cls, subfolder: str | None = None, file_name: str | None = None) -> BackendPlan:
        return cls("full", subfolder=subfolder, file_name=file_name)

    @property
    def is_full(self) -> bool:
        return self.mode == "full"

    def split_model_kwargs(
        self, backend: str, model_kwargs: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split ``model_kwargs`` into the ones for the runtime of the backend and the ones for the modules.

        Only the whole model runs in the backend: with another plan, all of ``model_kwargs`` is for the modules.
        """
        if not self.is_full:
            return {}, dict(model_kwargs or {})
        backend_kwargs, model_kwargs = split_backend_kwargs(model_kwargs)
        if backend == "openvino" and "device" in model_kwargs:
            backend_kwargs["device"] = model_kwargs.pop("device")
        return backend_kwargs, model_kwargs


@dataclass(frozen=True)
class _ExportRequest:
    """What ``model_kwargs`` asks for an export of the whole model."""

    export: bool = False
    subfolder: str = ""
    file_name: str | None = None

    @classmethod
    def from_model_kwargs(cls, model_kwargs: dict[str, Any]) -> _ExportRequest:
        return cls(
            export=bool(model_kwargs.get("export")),
            subfolder=model_kwargs.get("subfolder") or "",
            file_name=model_kwargs.get("file_name"),
        )

    def candidates(self, backend: str) -> list[Path]:
        """The directories that may hold the export, by order of preference."""
        if self.file_name is not None and len(Path(self.file_name).parts) > 1:
            return [Path(self.subfolder, *Path(self.file_name).parts[:-1])]
        return [Path(self.subfolder), Path(self.subfolder, backend)]


def _find_existing_export(
    model_name_or_path: str, candidates: list[Path], token: bool | str | None, revision: Any
) -> Path | None:
    """The first candidate directory, local or on the Hub, that holds the config of an export of the whole model."""
    load_path = Path(model_name_or_path)
    if load_path.exists():
        return next((c for c in candidates if (load_path / c / EXPORT_CONFIG_FILE).exists()), None)
    repo_files = set(
        list_repo_files(
            load_path.as_posix(),
            repo_type="model",
            revision=getattr(revision, "resolved", revision),
            token=token,
        )
    )
    return next((c for c in candidates if (c / EXPORT_CONFIG_FILE).as_posix() in repo_files), None)


def _is_legacy_export(
    model_name_or_path: str,
    backend: str,
    model_kwargs: dict[str, Any],
    token: bool | str | None,
    revision: Any,
) -> bool:
    """Whether the files to load were exported with Optimum, which only holds the Transformer model."""
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


def resolve_backend_plan(
    model_name_or_path: str,
    backend: str,
    model_kwargs: dict[str, Any] | None,
    token: bool | str | None = None,
    revision: Any = None,
) -> BackendPlan:
    """Decide how to load ``model_name_or_path`` with ``backend``.

    Raises an ``ImportError`` if the whole model has to be exported but cannot be. Warnings are not logged here: they
    are in ``BackendPlan.message``.
    """
    if backend == "torch":
        return BackendPlan.torch()

    model_kwargs = model_kwargs or {}
    backend_name = BACKEND_NAMES[backend]
    available, reason = is_transformers_export_available(backend)
    request = _ExportRequest.from_model_kwargs(model_kwargs)

    if not request.export:
        candidates = request.candidates(backend)
        subfolder = _find_existing_export(model_name_or_path, candidates, token, revision)
        if subfolder is not None:
            file_name = Path(request.file_name).name if request.file_name is not None else None
            return BackendPlan.load_full(subfolder.as_posix(), file_name)

        # Files exported with Optimum are loaded with Optimum
        if _is_legacy_export(model_name_or_path, backend, model_kwargs, token, revision):
            advice = (
                'To export the whole model instead, load the model with `model_kwargs={"export": True}` '
                "and save it with `save_pretrained()`."
                if available
                else f"To export the whole model instead, {reason}."
            )
            return BackendPlan.legacy(
                f"The {backend_name} model in {model_name_or_path!r} was exported with Optimum, so it only contains the "
                f"Transformer model: the other modules (e.g. pooling and normalization) run in PyTorch. {advice}"
            )

    if not available:
        raise ImportError(
            f"No {backend_name} model was found in {model_name_or_path!r} to load, and it cannot be exported: {reason}."
        )
    return BackendPlan.export_full(f"Exporting the whole model to {backend_name}.")


def split_backend_kwargs(model_kwargs: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``model_kwargs`` into the ones for the backend runtime and the ones for ``from_pretrained``."""
    model_kwargs = dict(model_kwargs or {})
    backend_kwargs = {key: model_kwargs.pop(key) for key in BACKEND_MODEL_KWARGS if key in model_kwargs}
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
