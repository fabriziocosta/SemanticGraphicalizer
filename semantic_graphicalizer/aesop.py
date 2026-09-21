"""Download, cache, and split the Project Gutenberg Aesop text."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from urllib.request import Request, urlopen


AESOP_GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/53103/pg53103.txt"
DEFAULT_AESOP_CACHE_DIR = Path("data") / "raw"
DEFAULT_AESOP_CACHE_FILE = "pg53103.txt"
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


def _extract_stories(book_text: str, limit: int) -> list[str]:
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
        if len(stories) == limit:
            break
    return stories


def load_aesop_fables(
    limit: int = 2,
    *,
    cache_dir: str | Path = DEFAULT_AESOP_CACHE_DIR,
    refresh: bool = False,
    url: str = AESOP_GUTENBERG_URL,
) -> list[str]:
    """Return the first ``limit`` Aesop stories as complete document strings.

    The Gutenberg source is downloaded once to ``cache_dir/pg53103.txt`` and
    reused on later calls. Set ``refresh=True`` to download it again.
    """

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")

    cache_path = Path(cache_dir) / DEFAULT_AESOP_CACHE_FILE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        book_text = _download_text(url, cache_path)
    else:
        book_text = cache_path.read_text(encoding="utf-8")
    return _extract_stories(book_text, limit)


__all__ = [
    "AESOP_GUTENBERG_URL",
    "DEFAULT_AESOP_CACHE_DIR",
    "load_aesop_fables",
]
