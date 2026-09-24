"""Download, cache, and split the Project Gutenberg Aesop text."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from urllib.request import Request, urlopen


AESOP_GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/53103/pg53103.txt"
DEFAULT_AESOP_CACHE_DIR = Path("data") / "raw"
DEFAULT_AESOP_CACHE_FILE = "pg53103.txt"
DEFAULT_AESOP_STORIES_CACHE_FILE = "aesop_fables.json"
_TITLE_PATTERN = re.compile(r"(?m)^\s{5,}([A-ZÆŒ][A-ZÆŒ'’& ,.-]{2,80})\s*$")


def _download_text(
    url: str,
    destination: Path,
    *,
    opener: Callable[..., object] | None = None,
) -> str:
    request = Request(url, headers={"User-Agent": "semantic-graphicalizer/0.1"})
    download_opener = opener or urlopen
    with download_opener(request, timeout=30) as response:  # type: ignore[union-attr]
        content = response.read()  # type: ignore[union-attr]
    text = content.decode("utf-8")
    destination.write_text(text, encoding="utf-8")
    return text


def _extract_stories(book_text: str, limit: int | None = None) -> list[str]:
    body = book_text.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[-1]
    body = body.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    headings = list(_TITLE_PATTERN.finditer(body))

    stories: list[str] = []
    for index, heading in enumerate(headings):
        title = heading.group(1).strip()
        if len(title.split()) < 2:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        story = body[heading.start():end].strip()
        if len(story) < 250:
            continue
        stories.append(story)
        if limit is not None and len(stories) == limit:
            break
    return stories


def _read_story_cache(path: Path) -> list[str] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, list) or not all(isinstance(story, str) for story in cached):
        return None
    return cached


def load_aesop_fables(
    limit: int | None = 2,
    *,
    cache_dir: str | Path = DEFAULT_AESOP_CACHE_DIR,
    refresh: bool = False,
    url: str = AESOP_GUTENBERG_URL,
    select_at_random: bool = False,
    rand_seed: int | None = None,
) -> list[str]:
    """Return up to ``limit`` Aesop stories as complete document strings.

    Set ``limit=None`` to return the complete cached collection.

    By default, stories are returned in source order. Set
    ``select_at_random=True`` to sample the requested number from the complete
    cached collection. ``rand_seed`` makes that sample reproducible; ``None``
    uses the standard nondeterministic random seed.

    The parsed stories are cached in ``cache_dir/aesop_fables.json``. The raw
    Gutenberg source is cached in ``cache_dir/pg53103.txt`` and is downloaded
    only when neither cache is available. Set ``refresh=True`` to rebuild both.
    """

    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if not isinstance(select_at_random, bool):
        raise ValueError("select_at_random must be a boolean")
    if rand_seed is not None and (not isinstance(rand_seed, int) or isinstance(rand_seed, bool)):
        raise ValueError("rand_seed must be an integer or None")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")

    def select_stories(stories: list[str]) -> list[str]:
        if limit is None:
            return stories
        if not select_at_random:
            return stories[:limit]
        sampler = random.Random(rand_seed)
        return sampler.sample(stories, k=min(limit, len(stories)))

    cache_path = Path(cache_dir) / DEFAULT_AESOP_CACHE_FILE
    stories_cache_path = Path(cache_dir) / DEFAULT_AESOP_STORIES_CACHE_FILE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not refresh:
        cached_stories = _read_story_cache(stories_cache_path)
        if cached_stories is not None:
            return select_stories(cached_stories)

    if refresh or not cache_path.exists():
        book_text = _download_text(url, cache_path)
    else:
        book_text = cache_path.read_text(encoding="utf-8")
    stories = _extract_stories(book_text)
    stories_cache_path.write_text(
        json.dumps(stories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return select_stories(stories)


__all__ = [
    "AESOP_GUTENBERG_URL",
    "DEFAULT_AESOP_CACHE_DIR",
    "DEFAULT_AESOP_STORIES_CACHE_FILE",
    "load_aesop_fables",
]
