"""Knowledge-base loading: markdown articles with YAML-ish front matter, chunked by section."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    intents: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_id: str
    article_title: str
    heading: str
    text: str

    @property
    def search_text(self) -> str:
        # Title and heading are repeated into the indexed text: short user queries often
        # match the topic ("refund status") better than the body wording.
        return f"{self.article_title}. {self.heading}. {self.text}"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError("KB article is missing front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def load_articles(kb_dir: Path) -> list[Article]:
    articles = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        intents = tuple(i.strip() for i in meta.get("intents", "").strip("[]").split(",") if i.strip())
        articles.append(Article(id=meta["id"], title=meta["title"], intents=intents, body=body.strip()))
    if not articles:
        raise FileNotFoundError(f"No KB articles found in {kb_dir}")
    return articles


def chunk_articles(articles: list[Article]) -> list[Chunk]:
    chunks = []
    for article in articles:
        sections = re.split(r"^##\s+", article.body, flags=re.MULTILINE)
        for n, section in enumerate(s for s in sections if s.strip()):
            heading, _, text = section.partition("\n")
            chunks.append(
                Chunk(
                    chunk_id=f"{article.id}#{n}",
                    article_id=article.id,
                    article_title=article.title,
                    heading=heading.strip(),
                    text=text.strip(),
                )
            )
    return chunks


def intent_to_articles(articles: list[Article]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for article in articles:
        for intent in article.intents:
            mapping.setdefault(intent, set()).add(article.id)
    return mapping
