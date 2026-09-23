from pathlib import Path

import pytest

from semantic_graphicalizer import load_ontology, load_prompts
from semantic_graphicalizer.exceptions import ConfigurationError


ROOT = Path(__file__).parents[1]


def test_aesop_configuration_loads_recursive_vocabularies() -> None:
    ontology = load_ontology(ROOT / "configs/ontologies/aesop.yaml")
    prompts = load_prompts(ROOT / "configs/prompts/aesop.yaml")
    assert ontology.name == "aesop-narrative"
    assert "Character" in ontology.term_ids
    assert "interacts_with" in ontology.relation_ids
    assert {"actor", "target"} <= ontology.argument_role_ids
    assert ontology.relation("interacts_with").projection == ("actor", "target")
    assert ontology.relation("causes").category == "causal"
    assert ontology.relation("before").category == "temporal"
    assert ontology.relation("before").projection == ("earlier", "later")
    assert prompts.render("extract", text="[]", ontology="demo")


def test_ontology_rejects_duplicate_terms() -> None:
    with pytest.raises(ConfigurationError, match="term IDs must be unique"):
        load_ontology({"name": "bad", "version": "1", "terms": [{"id": "Thing", "description": "one"}, {"id": "Thing", "description": "two"}], "relations": {}})


def test_ontology_rejects_unknown_relation_types_and_invalid_projection() -> None:
    with pytest.raises(ConfigurationError, match="unknown types"):
        load_ontology({"name": "bad", "version": "1", "terms": [{"id": "Thing", "description": "one"}], "relations": {"rel": {"description": "relation", "arguments": {"role": {"allowed_types": ["Missing"]}}}}})
    with pytest.raises(ConfigurationError, match="exactly two roles"):
        load_ontology({"name": "bad", "version": "1", "terms": [{"id": "Thing", "description": "one"}], "relations": {"rel": {"description": "relation", "arguments": {"role": {}}, "projection": ["role"]}}})
    with pytest.raises(ConfigurationError, match="category must be"):
        load_ontology({"name": "bad", "version": "1", "terms": [{"id": "Thing", "description": "one"}], "relations": {"rel": {"description": "relation", "category": "spatial"}}})


def test_prompts_require_recursive_extraction_stages() -> None:
    with pytest.raises(ConfigurationError, match="must contain exactly"):
        load_prompts({"domain": "bad", "version": "1", "stages": {}})
