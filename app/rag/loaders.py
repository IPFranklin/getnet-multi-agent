"""Loaders: curated markdown knowledge base + live web/PDF crawling."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup

from app.rag.chunking import Document

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; GetnetSupportBot/1.0; +https://github.com/)"
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def load_markdown_dir(directory: Path) -> list[Document]:
    """Curated docs with YAML front matter (title, url, tags, language)."""
    docs = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict = {}
        m = _FRONT_MATTER.match(raw)
        if m:
            meta = yaml.safe_load(m.group(1)) or {}
            raw = raw[m.end():]
        docs.append(
            Document(
                text=raw.strip(),
                url=meta.get("url", f"kb://{path.stem}"),
                title=meta.get("title", path.stem.replace("-", " ").title()),
                source_type="curated",
                tags=list(meta.get("tags", [])),
                language=meta.get("language", "pt"),
            )
        )
    return docs


def html_to_text(html: str) -> tuple[str, str]:
    """Extract (title, main_text) from an HTML page, dropping chrome (nav/footer/scripts)."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe"]):
        tag.decompose()
    for sel in ["[class*=cookie]", "[id*=cookie]", "[class*=menu]", "[class*=breadcrumb]"]:
        for tag in soup.select(sel):
            tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if len(text) < 3:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            lines.append(f"\n{'#' * int(el.name[1])} {text}\n")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    # de-duplicate repeated boilerplate lines while preserving order
    seen, out = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return title, "\n\n".join(out)


def pdf_to_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    return re.sub(r"[ \t]+", " ", text)


def fetch_document(url: str, client: httpx.Client, tags: list[str] | None = None, language: str = "pt") -> Document | None:
    try:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("fetch_failed", extra={"fields": {"url": url, "error": str(exc)}})
        return None
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        text, title, stype = pdf_to_text(resp.content), url.rsplit("/", 1)[-1], "pdf"
    else:
        title, text = html_to_text(resp.text)
        stype = "web"
    if len(text) < 200:
        logger.warning("fetch_too_little_text", extra={"fields": {"url": url, "chars": len(text)}})
        return None
    return Document(text=text, url=url, title=title or url, source_type=stype, tags=tags or [], language=language)


def load_sources(sources_file: Path) -> list[dict]:
    if not sources_file.exists():
        return []
    data = yaml.safe_load(sources_file.read_text(encoding="utf-8")) or {}
    return data.get("sources", [])


def crawl_sources(sources: list[dict], timeout: float = 20.0, transport: httpx.BaseTransport | None = None) -> list[Document]:
    docs = []
    with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, transport=transport) as client:
        for src in sources:
            doc = fetch_document(src["url"], client, src.get("tags"), src.get("language", "pt"))
            if doc:
                docs.append(doc)
    return docs
