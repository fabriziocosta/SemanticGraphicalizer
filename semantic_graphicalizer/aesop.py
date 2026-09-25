"""Download, cache, and split the Project Gutenberg Aesop text."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from urllib.request import Request, urlopen


AESOP_GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/21/pg21.txt"
DEFAULT_AESOP_CACHE_DIR = Path("data") / "raw"
DEFAULT_AESOP_CACHE_FILE = "pg21.txt"
DEFAULT_AESOP_STORIES_CACHE_FILE = "aesop_300_fables.json"
_TITLE_PATTERN = re.compile(r"(?m)^\s{5,}([A-ZÆŒ][A-ZÆŒ'’& ,.-]{2,80})\s*$")


def _normalized_heading(value: str) -> str:
    return " ".join(value.replace("’", "'").split()).casefold()


def _extract_indexed_fables(book_text: str) -> list[str] | None:
    """Extract the Townsend edition using its contents list as the heading index."""

    lines = book_text.splitlines()
    normalized = [_normalized_heading(line) for line in lines]
    contents_start = next((i for i, line in enumerate(normalized) if line == "contents"), None)
    if contents_start is None:
        return None

    fables_title = "aesop's fables"
    toc_start = next((i for i in range(contents_start + 1, len(lines)) if normalized[i] == fables_title), None)
    if toc_start is None:
        return None
    toc_end = next((i for i in range(toc_start + 1, len(lines)) if normalized[i] == "footnotes"), None)
    if toc_end is None:
        return None
    titles = [line.strip() for line in lines[toc_start + 1:toc_end] if line.strip() and _normalized_heading(line) != "index"]
    if not titles:
        return None

    story_start = next((i for i in range(toc_end + 1, len(lines)) if normalized[i] == fables_title), None)
    if story_start is None:
        return None
    story_end = next((i for i in range(story_start + 1, len(lines)) if normalized[i] == "footnotes"), None)
    if story_end is None:
        return None

    title_set = {_normalized_heading(title) for title in titles}
    headings = [
        (index, lines[index].strip())
        for index in range(story_start + 1, story_end)
        if normalized[index] in title_set
    ]
    if [_normalized_heading(title) for _, title in headings] != [
        _normalized_heading(title) for title in titles
    ]:
        return None

    return [
        "\n".join(lines[start:end]).strip()
        for (start, _), (end, _) in zip(headings, [*headings[1:], (story_end, "")])
    ]


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
    indexed_stories = _extract_indexed_fables(body)
    if indexed_stories is not None:
        return indexed_stories if limit is None else indexed_stories[:limit]

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

    The parsed stories are cached in ``cache_dir/aesop_300_fables.json``. The
    raw Gutenberg source is cached in ``cache_dir/pg21.txt`` and is downloaded
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
