"""Text normalisation shared by BM25 and the hash embedder (PT/EN aware)."""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS = set(
    """
    a o as os um uma uns umas de do da dos das em no na nos nas por para pra com sem e ou que
    se sua seu suas seus meu minha meus minhas eu voce voces ele ela eles elas isso esse essa
    este esta qual quais como quando onde porque ao aos ate mais menos muito ja nao sim ser
    estar ter tem sao foi era pelo pela pelos pelas me te lhe nos the an and or of to in on
    for with is are was were be been do does did can could i my me you your it its this that
    what which how when where why will would should from at by as about into than then there
    """.split()
)
_TOKEN = re.compile(r"[a-z0-9]+")

# Light PT/EN synonym normalisation so BM25 matches across languages and phrasings.
_SYNONYMS = {
    "maquina": "maquininha", "machine": "maquininha", "terminal": "maquininha", "pos": "maquininha",
    "installments": "parcelas", "installment": "parcelas", "parcelamento": "parcelas", "parcelar": "parcelas",
    "advance": "antecipacao", "anticipation": "antecipacao", "antecipar": "antecipacao",
    "receivables": "recebiveis", "recebivel": "recebiveis",
    "fees": "taxas", "fee": "taxas", "rates": "taxas", "taxa": "taxas",
    "rent": "aluguel", "rental": "aluguel",
    "link": "link", "payment": "pagamento", "payments": "pagamento", "pagamentos": "pagamento",
    "bank": "banco", "account": "conta", "deposit": "deposito", "deposited": "deposito",
    "decline": "recusada", "declined": "recusada", "negada": "recusada", "recusado": "recusada",
    "internet": "conexao", "connect": "conexao", "connection": "conexao", "wifi": "conexao", "conectar": "conexao",
    "classica": "classica", "classic": "classica", "smart": "smart",
    "whatsapp": "whatsapp", "zap": "whatsapp",
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(strip_accents(text.lower()))
    return [_SYNONYMS.get(t, t) for t in tokens if t not in _STOPWORDS and len(t) > 1]
