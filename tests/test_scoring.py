"""Tests for the pure-Python scoring logic (no torch/model needed) — the code
path that turns per-token readouts into the headline hallucination score."""
from innerlens.core import InnerLens, TokenTrace, WorkspaceResult
from innerlens.lenses import lens_for
import pytest


def tok(token, conf, *, entity=True, content=True, out=float("nan")):
    return TokenTrace(token=token, token_id=0, internal_confidence=conf, entropy=0.0,
                      inner_monologue=[], is_content=content, is_entity=entity,
                      output_confidence=out)


def test_confidence_is_weakest_entity_token():
    r = WorkspaceResult("The Paris", [tok("The", 0.1, entity=False), tok(" Paris", 0.9)])
    assert r.confidence == 0.9          # function word "The" excluded from the score
    assert not r.likely_hallucinating


def test_flags_when_an_entity_is_unsupported():
    r = WorkspaceResult("Zorbia", [tok(" Made", 0.8), tok("upville", 0.12)])
    assert r.confidence == 0.12
    assert r.likely_hallucinating       # below default 0.5 threshold


def test_first_sentence_only():
    # a low-support entity AFTER the answer's first sentence must not drag the score
    toks = [tok(" Canberra", 0.9), tok(".", 0.5, entity=False, content=False),
            tok(" Located", 0.05)]      # trailing rambling
    r = WorkspaceResult("Canberra. Located...", toks)
    assert r.confidence == 0.9          # stops at the sentence boundary


def test_mean_entity_confidence():
    r = WorkspaceResult("a b", [tok("Foo", 0.6), tok("Bar", 0.8)])
    assert abs(r.mean_entity_confidence - 0.7) < 1e-9


def test_falls_back_to_content_when_no_entities():
    r = WorkspaceResult("the", [tok("the", 0.3, entity=False)])
    assert r.confidence == 0.3          # no entity tokens -> use content


def test_entity_classifier():
    assert InnerLens._is_entity("Paris")
    assert InnerLens._is_entity(" Tokyo")
    assert not InnerLens._is_entity("the")
    assert not InnerLens._is_entity(" of")
    assert not InnerLens._is_entity("   ")
    assert not InnerLens._is_entity(".")


def test_content_classifier():
    assert InnerLens._is_content(" Paris")
    assert not InnerLens._is_content("   ")
    assert not InnerLens._is_content(" ...")


def test_to_dict_shape():
    r = WorkspaceResult("x", [tok("Foo", 0.42, out=0.9)])
    d = r.to_dict()
    assert d["text"] == "x"
    assert d["confidence"] == 0.42
    assert d["output_confidence"] == 0.9
    assert d["tokens"][0]["output_confidence"] == 0.9
    assert "likely_hallucinating" in d and isinstance(d["tokens"], list)


def test_output_confidence_same_aggregation_as_internal():
    # weakest ENTITY token, function words excluded — identical rule to `confidence`
    r = WorkspaceResult("The Paris", [tok("The", 0.1, entity=False, out=0.05),
                                      tok(" Paris", 0.9, out=0.8),
                                      tok(" France", 0.7, out=0.6)])
    assert r.output_confidence == 0.6
    assert r.confidence == 0.7


def test_output_confidence_skips_untraced_nan_tokens():
    r = WorkspaceResult("a b", [tok("Foo", 0.6, out=0.4), tok("Bar", 0.8)])  # Bar untraced
    assert r.output_confidence == 0.4


def test_registry_known_and_unknown():
    assert lens_for("Qwen/Qwen3.5-4B").repo == "neuronpedia/jacobian-lens"
    with pytest.raises(KeyError):
        lens_for("does/not-exist")
