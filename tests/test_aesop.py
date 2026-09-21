from pathlib import Path

import pytest

from semantic_graphicalizer import load_aesop_fables


def gutenberg_fixture() -> str:
    return """*** START OF THE PROJECT GUTENBERG EBOOK AESOP'S FABLES ***

                         INTRODUCTION

                         This is front matter.

                         THE FIRST FABLE

The first story has enough text to be treated as a complete tale. It describes
a character taking an action and learning a lesson. The story continues with
several sentences so the loader's minimum story length is satisfied. It ends
with a clear moral for the reader to consider.

                         THE SECOND FABLE

The second story has enough text to be treated as a complete tale. It describes
another character meeting a challenge and responding with a thoughtful action.
The story continues with several sentences so the loader returns it as a
separate document. It also ends with a clear lesson.

                         THE THIRD FABLE

The third story is not selected when the requested subset is two stories. It is
included to prove that the loader stops after the requested number.

*** END OF THE PROJECT GUTENBERG EBOOK AESOP'S FABLES ***"""


class FakeResponse:
    def __init__(self, content: str):
        self.content = content.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.content


def test_loader_returns_first_two_stories_and_caches_source(tmp_path, monkeypatch) -> None:
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(gutenberg_fixture())

    monkeypatch.setattr("semantic_graphicalizer.aesop.urlopen", opener)
    stories = load_aesop_fables(cache_dir=tmp_path)

    assert len(stories) == 2
    assert stories[0].splitlines()[0] == "THE FIRST FABLE"
    assert stories[1].splitlines()[0] == "THE SECOND FABLE"
    assert len(calls) == 1
    assert (tmp_path / "pg53103.txt").exists()

    def unexpected_download(*args, **kwargs):
        raise AssertionError("cache should be used")

    monkeypatch.setattr("semantic_graphicalizer.aesop.urlopen", unexpected_download)
    cached_stories = load_aesop_fables(cache_dir=tmp_path)
    assert cached_stories == stories


def test_loader_refreshes_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "semantic_graphicalizer.aesop.urlopen",
        lambda request, timeout: FakeResponse(gutenberg_fixture()),
    )
    load_aesop_fables(cache_dir=tmp_path)
    refreshed = load_aesop_fables(cache_dir=tmp_path, refresh=True)
    assert len(refreshed) == 2


def test_loader_rejects_invalid_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        load_aesop_fables(limit=0, cache_dir=tmp_path)
