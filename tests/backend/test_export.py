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
from sentence_transformers.base.modules import Transformer

SENTENCES = ["Hello, World!", "A much longer sentence about the weather today."]
PAIRS = [("Which planet is red?", "Mars is the red planet."), ("Which planet is red?", "Not Venus.")]

MODELS = {
    "sentence_transformer": (SentenceTransformer, "sentence-transformers-testing/stsb-bert-tiny-safetensors"),
    "cross_encoder": (CrossEncoder, "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"),
    "sparse_encoder": (SparseEncoder, "sparse-encoder-testing/splade-bert-tiny-nq"),
    "multi_vector_encoder": (MultiVectorEncoder, "sentence-transformers-testing/stsb-bert-tiny-safetensors"),
}
EXPORT_FORMATS = ["onnx", "openvino"]
EXPORT_PACKAGES = ["onnxscript", "openvino"]

# Exported by a previous version of sentence-transformers: only its Transformer module is exported
LEGACY_MODEL = "sentence-transformers-testing/stsb-bert-tiny-openvino"

requires_exporters = pytest.mark.skipif(
    not is_transformers_export_available() or any(importlib.util.find_spec(p) is None for p in EXPORT_PACKAGES),
    reason=f"exporting a model requires transformers>={MIN_TRANSFORMERS_VERSION} and {EXPORT_PACKAGES}",
)


def compute(model) -> list[np.ndarray]:
    if isinstance(model, CrossEncoder):
        return [np.asarray(model.predict(PAIRS))]
    if isinstance(model, SparseEncoder):
        return [model.encode(SENTENCES).to_dense().numpy()]
    if isinstance(model, MultiVectorEncoder):
        return [np.asarray(embedding) for embedding in model.encode(SENTENCES)]
    return [model.encode(SENTENCES)]


def export_model(architecture, export_format, tmp_path):
    """Export a model, and return it with outputs and where it is saved"""
    model_class, model_id = MODELS[architecture]
    outputs = compute(model_class(model_id))
    model = model_class(model_id, backend=export_format)
    model.save(str(tmp_path / "export"))
    return model, outputs, tmp_path / "export"


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
@pytest.mark.parametrize("architecture", MODELS)
def test_exported_model_matches(architecture, export_format, tmp_path):
    model, expected_output, _ = export_model(architecture, export_format, tmp_path)
    assert model.get_backend() == export_format
    outputs = compute(model)
    for expected_output, output in zip(expected_output, outputs):
        assert np.allclose(expected_output, output, atol=1e-3)


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_saved_export_is_loaded_not_exported_again(export_format, tmp_path, monkeypatch):
    _, expected_output, directory = export_model("sentence_transformer", export_format, tmp_path)
    assert has_export_config(directory)

    reloaded = SentenceTransformer(str(directory), backend=export_format)
    outputs = compute(reloaded)
    for expected_output, output in zip(expected_output, outputs):
        assert np.allclose(expected_output, output, atol=1e-3)


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
    assert has_export_config(tmp_path)
