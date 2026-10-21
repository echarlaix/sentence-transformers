from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.backend.export import (
    MIN_TRANSFORMERS_VERSION,
    has_export_config,
    is_transformers_export_available,
)

SENTENCES = ["Hello, World!", "A much longer sentence about the weather today."]
PAIRS = [("Which planet is red?", "Mars is the red planet."), ("Which planet is red?", "Not Venus.")]

MODELS = {
    "sentence_transformer": (SentenceTransformer, "sentence-transformers-testing/stsb-bert-tiny-safetensors"),
    "cross_encoder": (CrossEncoder, "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"),
    "sparse_encoder": (SparseEncoder, "sparse-encoder-testing/splade-bert-tiny-nq"),
    "multi_vector_encoder": (MultiVectorEncoder, "multi-vector-encoder-testing/bert-tiny-multi-vector"),
}
EXPORT_FORMATS = ["onnx", "openvino"]
EXPORT_PACKAGES = ["onnxscript", "openvino"]
# OpenVINO runs in the precision of the hardware, which can be lower than float32
ATOL = {"onnx": 1e-3, "openvino": 1e-1}

# Exported by a previous version of sentence-transformers: only its Transformer module is exported
LEGACY_MODEL = "sentence-transformers-testing/stsb-bert-tiny-openvino"

requires_exporters = pytest.mark.skipif(
    not is_transformers_export_available() or any(importlib.util.find_spec(p) is None for p in EXPORT_PACKAGES),
    reason=f"exporting a model requires transformers>={MIN_TRANSFORMERS_VERSION} and {EXPORT_PACKAGES}",
)


def has_quantizer(name: str, packages: list[str]) -> bool:
    if not is_transformers_export_available() or any(importlib.util.find_spec(p) is None for p in packages):
        return False
    import transformers.exporters

    return hasattr(transformers.exporters, name)


requires_onnxruntime_quantizer = pytest.mark.skipif(
    not has_quantizer("OnnxRuntimeQuantizer", [*EXPORT_PACKAGES, "onnxruntime"]),
    reason="quantizing with ONNX Runtime requires a transformers version with OnnxRuntimeQuantizer, and onnxruntime",
)
requires_nncf_onnx_quantizer = pytest.mark.skipif(
    not has_quantizer("NNCFOnnxQuantizer", [*EXPORT_PACKAGES, "onnxruntime", "nncf"]),
    reason="quantizing with NNCF requires a transformers version with NNCFOnnxQuantizer, onnxruntime and nncf",
)
requires_nncf_openvino_quantizer = pytest.mark.skipif(
    not has_quantizer("NNCFOpenVINOQuantizer", [*EXPORT_PACKAGES, "nncf"]),
    reason="quantizing with NNCF requires a transformers version with NNCFOpenVINOQuantizer and nncf",
)

QUANTIZED_ONNX_OPS = {"QuantizeLinear", "DynamicQuantizeLinear", "MatMulInteger", "QLinearMatMul", "MatMulNBits"}


def compute(model) -> list[np.ndarray]:
    if isinstance(model, CrossEncoder):
        return [np.asarray(model.predict(PAIRS))]
    if isinstance(model, SparseEncoder):
        return [model.encode(SENTENCES).to_dense().numpy()]
    if isinstance(model, MultiVectorEncoder):
        return [np.asarray(embedding) for embedding in model.encode(SENTENCES)]
    return [model.encode(SENTENCES)]


def export_model(architecture, export_format, tmp_path, export_config=None):
    """Export a model, optionally with an export config (with a quantizer) and save it. Returns the model and
    the directory it is saved to"""
    model_class, model_id = MODELS[architecture]
    model_kwargs = {"export_config": export_config} if export_config is not None else {}
    model = model_class(model_id, backend=export_format, model_kwargs=model_kwargs)
    model.save(str(tmp_path / "export"))
    return model, tmp_path / "export"


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
@pytest.mark.parametrize("architecture", MODELS)
def test_exported_model_matches(architecture, export_format, tmp_path):
    model_class, model_id = MODELS[architecture]
    expected = compute(model_class(model_id))
    model, _ = export_model(architecture, export_format, tmp_path)
    assert model.get_backend() == export_format
    for expected_output, output in zip(expected, compute(model)):
        assert np.allclose(expected_output, output, atol=ATOL[export_format])


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_saved_export_is_loaded_not_exported_again(export_format, tmp_path, monkeypatch):
    expected = compute(SentenceTransformer(MODELS["sentence_transformer"][1]))
    _, directory = export_model("sentence_transformer", export_format, tmp_path)
    assert has_export_config(directory, export_format)

    reloaded = SentenceTransformer(str(directory), backend=export_format)
    for expected_output, output in zip(expected, compute(reloaded)):
        assert np.allclose(expected_output, output, atol=ATOL[export_format])


def test_legacy_export_is_loaded():
    pytest.importorskip("optimum.intel")
    model = SentenceTransformer(LEGACY_MODEL, backend="openvino")
    assert model._exported_model is None
    assert type(model[0].model).__name__.startswith("OV")
    for expected_output, output in zip(compute(SentenceTransformer(LEGACY_MODEL)), compute(model)):
        assert np.allclose(expected_output, output, atol=5e-2)


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_legacy_export_is_exported_again(export_format, tmp_path):
    expected = compute(SentenceTransformer(LEGACY_MODEL))
    model = SentenceTransformer(LEGACY_MODEL, backend=export_format, model_kwargs={"export": True})
    assert model._exported_model is not None
    for expected_output, output in zip(expected, compute(model)):
        assert np.allclose(expected_output, output, atol=5e-2)

    model.save(str(tmp_path))
    assert has_export_config(tmp_path, export_format)


def onnx_op_types(directory) -> set[str]:
    import onnx

    (path,) = directory.glob("*.onnx")
    return {node.op_type for node in onnx.load(path, load_external_data=False).graph.node}


def openvino_op_types(directory) -> set[str]:
    import openvino

    (path,) = directory.glob("*.xml")
    operations = openvino.Core().read_model(path).get_ops()
    types = {operation.get_type_name() for operation in operations}
    for operation in operations:
        if operation.get_type_name() == "Constant":
            types.add(operation.get_output_element_type(0).get_type_name())
    return types


@requires_onnxruntime_quantizer
@pytest.mark.parametrize("architecture", MODELS)
@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_onnxruntime_quantized_export(architecture, mode, tmp_path):
    from transformers.exporters import (
        OnnxConfig,
        OnnxRuntimeDynamicConfig,
        OnnxRuntimeQuantizer,
        OnnxRuntimeStaticConfig,
    )

    config = OnnxRuntimeDynamicConfig() if mode == "dynamic" else OnnxRuntimeStaticConfig()
    export_config = OnnxConfig(dynamic=True, quantizer=OnnxRuntimeQuantizer(config))
    model, directory = export_model(architecture, export_config.export_format.value, tmp_path, export_config)

    assert model.get_backend() == "onnx"
    assert onnx_op_types(directory) & QUANTIZED_ONNX_OPS


@requires_nncf_onnx_quantizer
@pytest.mark.parametrize("architecture", MODELS)
def test_nncf_quantized_onnx_export(architecture, tmp_path):
    from transformers.exporters import NNCFOnnxQuantizer, OnnxConfig

    export_config = OnnxConfig(dynamic=True, quantizer=NNCFOnnxQuantizer())
    model, directory = export_model(architecture, export_config.export_format.value, tmp_path, export_config)

    assert model.get_backend() == "onnx"
    assert onnx_op_types(directory) & QUANTIZED_ONNX_OPS


@requires_nncf_openvino_quantizer
@pytest.mark.parametrize("architecture", MODELS)
@pytest.mark.parametrize("mode", ["quantization", "weight_compression"])
def test_nncf_quantized_openvino_export(architecture, mode, tmp_path):
    from transformers.exporters import NNCFOpenVINOQuantizer, NNCFWeightCompressionConfig, OpenVINOConfig

    config = NNCFWeightCompressionConfig() if mode == "weight_compression" else None
    export_config = OpenVINOConfig(dynamic=True, quantizer=NNCFOpenVINOQuantizer(config))
    model, directory = export_model(architecture, export_config.export_format.value, tmp_path, export_config)

    assert model.get_backend() == "openvino"
    op_types = openvino_op_types(directory)
    assert "FakeQuantize" in op_types if mode == "quantization" else {"i8", "u8"} & op_types
