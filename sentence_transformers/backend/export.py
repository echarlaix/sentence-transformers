from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
import torch
import transformers
from packaging import version
from torch import nn
from transformers.utils import logging as transformers_logging

if TYPE_CHECKING:
    from transformers.exporters import ExportArtifacts, ExportConfigMixin, ExportedModel

    from sentence_transformers.base.model import BaseModel

logger = transformers_logging.get_logger(__name__)

# first version to include `XxxExporter`, `XxxModelRunner`, XxxQuantizer` is v5.20
MIN_TRANSFORMERS_VERSION = "5.19.0.dev0"
LEGACY_EXPORT_FORMATS = ["onnx", "openvino"]
EXPORT_MANIFEST_FILE = "export.json"

# Sample texts used to trace the model; two different lengths so that the sequence dimension isn't specialized
_SAMPLE_TEXTS = ["Hello world!", "A somewhat longer sentence, so that the sequence dimension is not constant."]


def is_transformers_export_available() -> bool:
    """Whether the installed ``transformers`` can export a whole model and run the export."""
    installed = version.parse(transformers.__version__)
    return installed.release >= version.parse(MIN_TRANSFORMERS_VERSION).release


def _import_exporters():
    if not is_transformers_export_available():
        raise ImportError(
            f"Exporting or loading the whole model with the 'onnx' or 'openvino' backend requires "
            f"`transformers>={MIN_TRANSFORMERS_VERSION}` (found {transformers.__version__}). Models exported by "
            "previous versions of sentence-transformers can still be loaded."
        )
    try:
        import transformers.exporters as exporters
    except ImportError as exc:
        raise ImportError(
            "Exporting or loading a model with the 'onnx' or 'openvino' backend requires the dependencies of the "
            "chosen format, e.g. `pip install onnx onnxscript onnxruntime` for ONNX or `pip install openvino` for "
            "OpenVINO."
        ) from exc
    return exporters


def get_export_formats() -> list[str]:
    """Return the names of the export formats registered in ``transformers.exporters``, e.g. ``["onnx", ...]``.

    Without the exporters (``transformers`` is too old), only the formats of the legacy exports are returned.
    """
    if not is_transformers_export_available():
        return LEGACY_EXPORT_FORMATS
    return sorted(_import_exporters().EXPORT_BACKENDS)


class _ExportableModel(nn.Module):
    """
    Wraps a model so that it can be traced: tensors in, a flat dictionary of the newly computed tensors out.

    Non-tensor features (e.g. ``modality``) are not part of the graph. The inputs of the graph are the tensors that
    were passed at export time, so the sample inputs decide which features the exported model accepts.
    """

    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, **features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(dict(features))
        # Inputs that pass through are not outputs, but a module may replace one (e.g. the attention mask)
        return {
            key: value
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value is not features.get(key)
        }


def _sample_features(model: BaseModel) -> dict[str, torch.Tensor]:
    """Preprocess a small batch of texts (or text pairs for cross-encoders) to trace the model with."""
    inputs = [(text, text) for text in _SAMPLE_TEXTS] if model.model_type == "CrossEncoder" else _SAMPLE_TEXTS
    features = model.preprocess(inputs)
    return {key: value.to(model.device) for key, value in features.items() if isinstance(value, torch.Tensor)}


def _resolve_export_config(export_format: str, export_config: ExportConfigMixin | dict[str, Any] | None):
    from transformers.exporters import export_backend

    config_class = export_backend(export_format, "config")
    if export_config is None:
        # Dynamic shapes, as texts of any batch size and length must be supported
        return config_class(dynamic=True)
    if isinstance(export_config, dict):
        return config_class(**{"dynamic": True, **export_config})
    return export_config


def export_model(
    model: BaseModel,
    export_format: str,
    export_config: ExportConfigMixin | dict[str, Any] | None = None,
    sample_features: dict[str, torch.Tensor] | None = None,
) -> ExportArtifacts:
    """
    Export all modules of a model as a single graph with ``transformers.exporters``.

    Args:
        model: The :class:`~sentence_transformers.SentenceTransformer`, :class:`~sentence_transformers.CrossEncoder`,
            :class:`~sentence_transformers.SparseEncoder` or :class:`~sentence_transformers.MultiVectorEncoder` to export.
        export_format: The export format, i.e. one of the formats registered in ``transformers.exporters``
            such as ``"onnx"`` or ``"openvino"``.
        export_config: The export configuration, either an export config object (e.g.
            :class:`~transformers.exporters.OnnxConfig`) or a dictionary of its arguments. Defaults to an export
            config with dynamic shapes.
        sample_features: The tensors to trace the model with, as returned by ``model.preprocess(...)``. Defaults to
            the preprocessed output of two short texts (or text pairs for a cross-encoder).

    Returns:
        ExportArtifacts: The exported graph. Save it with ``save_pretrained``.
    """
    exporters = _import_exporters()
    config = _resolve_export_config(export_format, export_config)
    exporter = exporters.AutoHfExporter.from_config(config)
    if sample_features is None:
        sample_features = _sample_features(model)

    return exporter.export(_ExportableModel(model), sample_features, config)


class ExportedModule:
    """
    Runs an exported graph in place of the forward pass of a model: it takes the features dictionary produced by
    ``preprocess`` and returns it updated with the outputs of the graph, like the modules of a model do.
    """

    def __init__(
        self,
        model: ExportedModel,
        export_format: str,
        artifacts: ExportArtifacts | None = None,
        source: tuple[str, dict[str, Any]] | None = None,
    ):
        self.model = model
        self.export_format = export_format
        # Set when the model was exported just now, saving writes these
        self.artifacts = artifacts
        # Set when the model was loaded: the ``(model_name_or_path, download_kwargs)`` to copy the files from
        self.source = source

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __call__(self, features: dict[str, Any], **kwargs) -> dict[str, Any]:
        if kwargs:
            logger.warning_once(
                f"The forward keyword arguments {sorted(kwargs)} are ignored by the {self.export_format!r} backend, as an "
                "exported graph is traced without them. Modules whose behaviour depends on them, e.g. the query/document "
                "`task` of a MultiVectorEncoder, behave as if they were not set."
            )
        runner = self.model.runner
        tensors = {
            key: value
            for key, value in features.items()
            if isinstance(value, torch.Tensor) and runner.declares(key, value)
        }
        output_device = next(iter(tensors.values())).device if tensors else runner.device
        outputs = self.model(**{key: value.to(runner.device) for key, value in tensors.items()})
        for key, value in outputs.items():
            features[key] = value.to(output_device) if isinstance(value, torch.Tensor) else value
        return features

    def save(self, output_path: str | os.PathLike) -> None:
        """Save the exported graph and its manifest to ``output_path``."""
        if self.artifacts is not None:
            self.artifacts.save_pretrained(output_path)
            return

        from transformers.exporters.base import read_export_manifest, resolve_export_file

        model_name_or_path, download_kwargs = self.source
        manifest = read_export_manifest(model_name_or_path, **download_kwargs)
        files = [EXPORT_MANIFEST_FILE] + [entry["file"] for entry in manifest["components"].values()]
        # Large ONNX models keep their weights in files next to the graph, and OpenVINO has a .bin next to the .xml
        os.makedirs(output_path, exist_ok=True)
        for file_name in files:
            path = resolve_export_file(model_name_or_path, file_name, **download_kwargs)
            if path is None:
                continue
            shutil.copyfile(path, Path(output_path) / file_name)
            for sibling in Path(path).parent.glob(f"{Path(file_name).stem}*"):
                if sibling.name != Path(file_name).name and sibling.suffix in (".bin", ".data"):
                    shutil.copyfile(sibling, Path(output_path) / sibling.name)


def _download_kwargs(export_format: str, **load_kwargs) -> dict[str, Any]:
    kwargs = {
        "cache_dir": load_kwargs.get("cache_folder"),
        "revision": load_kwargs.get("revision"),
        "token": load_kwargs.get("token"),
        "local_files_only": load_kwargs.get("local_files_only", False),
        "subfolder": export_format,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def is_transformers_exported_model(model_name_or_path: str, export_format: str, **load_kwargs) -> bool:
    """
    Whether ``model_name_or_path`` has an export of the whole model in its ``export_format`` subfolder.

    It is told by the manifest next to the graph, which the exports of previous sentence-transformers versions don't have.
    """
    kwargs = _download_kwargs(export_format, **load_kwargs)
    manifest = Path(kwargs["subfolder"], EXPORT_MANIFEST_FILE)
    if Path(model_name_or_path).is_dir():
        return (Path(model_name_or_path) / manifest).is_file()
    revision = getattr(kwargs.get("revision"), "resolved", kwargs.get("revision"))
    if kwargs.get("local_files_only"):
        cached = huggingface_hub.try_to_load_from_cache(
            model_name_or_path, manifest.as_posix(), cache_dir=kwargs.get("cache_dir"), revision=revision
        )
        return isinstance(cached, str)
    try:
        return huggingface_hub.file_exists(
            model_name_or_path,
            manifest.as_posix(),
            revision=revision,
            token=kwargs.get("token"),
        )
    except huggingface_hub.errors.HfHubHTTPError:
        return False


def load_exported_model(
    model_name_or_path: str, export_format: str, device: str | torch.device | None = None, **load_kwargs
) -> ExportedModule | None:
    """
    Load the whole-model export in the ``export_format`` subfolder of a local directory or Hugging Face repository.

    Args:
        model_name_or_path: Local directory or Hub model ID.
        export_format: The export format, which is also the name of the subfolder, e.g. ``"onnx"``.
        device: The device the exported graph should run on.
        **load_kwargs: ``cache_folder``, ``revision``, ``token`` and ``local_files_only``.

    Returns:
        ExportedModule | None: The loaded model, or None if there is no export to load.
    """
    if not is_transformers_exported_model(model_name_or_path, export_format, **load_kwargs):
        return None
    exporters = _import_exporters()

    from transformers.exporters.base import load_export_runners

    download_kwargs = _download_kwargs(export_format, **load_kwargs)
    runners = load_export_runners(model_name_or_path, device=device, **download_kwargs)
    if len(runners) != 1:
        raise ValueError(
            f"Expected a single exported graph in the {export_format!r} subfolder of {model_name_or_path!r}, "
            f"but found {len(runners)}: {sorted(runners)}."
        )
    return ExportedModule(
        exporters.ExportedModel(next(iter(runners.values()))),
        export_format,
        source=(model_name_or_path, download_kwargs),
    )
