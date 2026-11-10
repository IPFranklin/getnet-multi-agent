import httpx

from app.rag.chunking import Document, chunk_document
from app.rag.embeddings import HashEmbedder
from app.rag.index import HybridIndex
from app.rag.ingest import build_index, dedupe
from app.rag.loaders import crawl_sources, html_to_text, load_markdown_dir
from app.rag.retriever import KnowledgeRetriever
from app.rag.text import tokenize


def test_tokenize_normalises_accents_stopwords_and_synonyms():
    assert tokenize("Como funciona a Antecipação?") == ["funciona", "antecipacao"]
    assert "maquininha" in tokenize("card machine")
    assert "parcelas" in tokenize("installments")


def test_chunking_respects_sections_and_size():
    text = "# A\n\n" + ("palavra " * 300) + "\n\n# B\n\nconteúdo curto"
    chunks = chunk_document(Document(text=text, url="u", title="T"), size=500, overlap=50)
    assert {c.section for c in chunks} == {"A", "B"}
    assert all(len(c.text) <= 1000 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_dedupe_removes_identical_text():
    d = Document(text="mesmo texto", url="a", title="T")
    d2 = Document(text="mesmo   TEXTO", url="b", title="T")
    assert len(dedupe(chunk_document(d) + chunk_document(d2))) == 1


def test_markdown_loader_reads_front_matter(settings):
    docs = load_markdown_dir(settings.knowledge_base_dir)
    assert len(docs) >= 8
    pix = next(d for d in docs if "Pix" in d.title)
    assert pix.url.startswith("https://") and "pix" in pix.tags and pix.source_type == "curated"


def test_html_to_text_drops_chrome():
    html = """<html><head><title>Get Smart</title><script>x()</script></head><body>
      <nav>Menu Home Produtos</nav><main><h1>Get Smart</h1><p>Maquininha Android com tela touchscreen.</p>
      <ul><li>Wi-Fi e 4G</li></ul></main><footer>Copyright</footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "Get Smart"
    assert "Android" in text and "- Wi-Fi e 4G" in text
    assert "Menu Home" not in text and "Copyright" not in text


def test_crawler_handles_success_and_failure():
    body = "<html><title>Pix</title><body><main>" + "".join(f"<p>O Pix cai na conta em segundos, item {i}.</p>" for i in range(20)) + "</main></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if "ok" in str(request.url):
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})
        return httpx.Response(500)

    docs = crawl_sources([{"url": "https://x.test/ok", "tags": ["pix"]}, {"url": "https://x.test/fail"}],
                         transport=httpx.MockTransport(handler))
    assert len(docs) == 1 and docs[0].tags == ["pix"]


def test_index_persists_and_reloads(settings):
    emb = HashEmbedder()
    index = build_index(settings, emb)
    loaded = HybridIndex.load(settings.index_dir, emb)
    assert loaded is not None and len(loaded.chunks) == len(index.chunks) > 0
    # different embedder -> refuse stale index
    assert HybridIndex.load(settings.index_dir, HashEmbedder(dim=128)) is None


def _retriever(settings) -> KnowledgeRetriever:
    return KnowledgeRetriever(build_index(settings, HashEmbedder()), top_k=5, min_score=settings.retrieval_min_score)


def test_retrieval_finds_the_right_documents(settings):
    r = _retriever(settings)
    cases = {
        "diferença entre Get Clássica e Get Smart": "Maquininhas",
        "quantas parcelas no crediário": "Crediário",
        "preciso de conta bancária para receber Pix": "Pix",
        "como funciona a antecipação de recebíveis": "Antecipação",
        "vender pelo WhatsApp com link de pagamento": "Link de pagamento",
        "maquininha não conecta na internet": "conexão",
    }
    for query, expected_title in cases.items():
        hits = r.retrieve(query)
        assert hits, query
        assert expected_title.lower() in hits[0].title.lower(), (query, [h.title for h in hits])


def test_irrelevant_query_returns_nothing(settings):
    assert _retriever(settings).retrieve("previsão do tempo em Porto Alegre amanhã") == []


def test_context_format_has_citations(settings):
    hits = _retriever(settings).retrieve("crediário parcelas")
    ctx = KnowledgeRetriever.format_context(hits)
    assert ctx.startswith("[1]") and "URL:" in ctx
    assert KnowledgeRetriever.format_context([]) == "NO_RELEVANT_DOCUMENTS_FOUND"
