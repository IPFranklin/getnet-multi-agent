"""Document model and structure-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field


@dataclass
class Document:
    text: str
    url: str
    title: str
    source_type: str = "web"  # web | pdf | curated
    tags: list[str] = field(default_factory=list)
    language: str = "pt"


@dataclass
class Chunk:
    id: str
    text: str
    url: str
    title: str
    section: str
    source_type: str
    tags: list[str]
    language: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embedding_text(self) -> str:
        # Prefixing title + section gives the embedding the chunk's context.
        return f"{self.title} — {self.section}\n{self.text}"


_HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)


def _split_sections(text: str, default_title: str) -> list[tuple[str, str]]:
    """Split markdown-ish text by headings, keeping the heading as section name."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(default_title, text)]
    sections = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append((default_title, text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(2).strip(), body))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs/sentences into windows of ~size chars with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= size:
            units.append(p)
        else:  # very long paragraph: split by sentence
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())

    chunks, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start overlap on a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail}\n\n{unit}" if tail else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    # hard-split pathological units without whitespace
    out = []
    for c in chunks:
        while len(c) > size * 2:
            out.append(c[:size])
            c = c[size - overlap:]
        out.append(c)
    return out


def chunk_document(doc: Document, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _split_sections(doc.text, doc.title):
        for piece in _window(body, size, overlap):
            digest = hashlib.sha1(f"{doc.url}|{section}|{piece}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=digest, text=piece, url=doc.url, title=doc.title, section=section,
                    source_type=doc.source_type, tags=list(doc.tags), language=doc.language,
                )
            )
    return chunks
