from pathlib import Path

import pytest

from semantic_graphicalizer import load_ontology, load_prompts
from semantic_graphicalizer.exceptions import ConfigurationError


ROOT = Path(__file__).parents[1]


def test_aesop_configuration_loads() -> None:
    ontology = load_ontology(ROOT / "configs/ontologies/aesop.yaml")
    prompts = load_prompts(ROOT / "configs/prompts/aesop.yaml")
    assert ontology.name == "aesop-narrative"
    assert "Character" in ontology.term_ids
    assert "interacts_with" in ontology.relation_ids
    assert prompts.render("summarize", text="A tale.", ontology="demo")


def test_ontology_rejects_duplicate_terms() -> None:
    with pytest.raises(ConfigurationError, match="term IDs must be unique"):
        load_ontology({
            "name": "bad",
            "version": "1",
            "terms": [
                {"id": "Thing", "description": "one"},
                {"id": "Thing", "description": "two"},
            ],
            "relations": [],
        })


def test_ontology_rejects_unknown_relation_terms() -> None:
    with pytest.raises(ConfigurationError, match="unknown terms"):
        load_ontology({
            "name": "bad",
            "version": "1",
            "terms": [{"id": "Thing", "description": "one"}],
            "relations": [{
                "id": "rel",
                "description": "relation",
                "source_terms": ["Missing"],
            }],
        })


def test_prompts_require_all_pipeline_stages() -> None:
    with pytest.raises(ConfigurationError, match="must contain exactly"):
        load_prompts({"domain": "bad", "version": "1", "stages": {}})
