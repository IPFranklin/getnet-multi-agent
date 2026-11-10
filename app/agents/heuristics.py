"""Deterministic keyword heuristics.

Used in two places:
  * as the **fallback router** when the LLM router fails/times out (reliability);
  * by the offline ``FakeChatModel`` (demo mode and tests).
"""

from __future__ import annotations

import re

from app.guardrails import normalize

_PT = {"que", "como", "minha", "meu", "minhas", "meus", "nao", "voce", "qual", "quando", "quanto", "posso",
       "preciso", "maquininha", "vendas", "dinheiro", "obrigado", "ola", "oi", "esta", "estou", "da", "do", "das",
       "dos", "para", "cair", "cai", "caiu", "hoje", "ontem", "eu", "recebi", "receber", "vou", "uma", "um",
       "escreva", "sobre", "quem", "ganhou", "porque", "por", "isso", "tem", "sao", "conta", "entre",
       "diferenca", "e", "o", "os", "as", "ativo", "ativar"}
_ES = {"como", "mi", "cuando", "puedo", "necesito", "dinero", "ventas", "gracias", "hola", "esta", "maquina",
       "ayer", "hoy", "el", "los", "las", "cuanto", "recibi", "escribe", "sobre", "quien", "gano", "mis", "por"}
_EN = {"help", "hi", "hello", "hey", "the", "what", "how", "my", "when", "can", "do", "is", "will", "does", "i",
       "you", "from", "yesterday", "today", "with", "an", "about", "of", "to", "for", "write", "who", "won",
       "why", "which", "where", "did", "receive", "received", "money", "sales", "machine", "please", "me"}


def detect_language(text: str) -> str:
    """Cheap language guess used only when the LLM router is unavailable.

    Falls back to Portuguese (the main market) only when no word is recognised at all.
    """
    words = set(re.findall(r"[a-z]+", normalize(text)))
    scores = {"pt": len(words & _PT), "es": len(words & _ES) - 0.5, "en": len(words & _EN)}
    lang = max(scores, key=scores.get)
    return lang if scores[lang] > 0 else "pt"


ESCALATION_TERMS = [
    r"\b(human|humano|atendente|attendant|agent[e]?|pessoa|person|operador|operator|gerente|manager)\b.*\b(talk|speak|falar|hablar|quero|want|need|preciso)\b",
    r"\b(talk|speak|falar|hablar)\b.*\b(human|humano|atendente|person|pessoa|someone|alguem)\b",
    r"\b(reclama[cç][aã]o|complaint|procon|reclame aqui|processar|lawyer|advogado)\b",
    r"\b(fraude|fraud|roubad[ao]|stolen|furtad[ao]|n[aã]o reconhe[cç]o|unrecognized|did not make)\b",
]
SUPPORT_TERMS = [
    r"\b(my|minha|meu|mis?|minhas|meus)\b.*\b(sales|vendas|ventas|money|dinheiro|dinero|deposit|payout|repasse|recebimento|recebiveis|machine|maquin|terminal|pos|account|conta|plano|plan|aluguel|rent|transa)",
    r"\b(deposited|depositad|cair na conta|cai na conta|caiu|when will .* (money|paid)|quando (vou|vai) receber|quando .* cai)\b",
    r"\b(won'?t connect|not connecting|n[aã]o conecta|sem internet|offline|n[aã]o liga|won'?t turn on)\b",
    r"\b(decline|declined|recusad|negad|rechazad|error code|c[oó]digo de erro|erro\b|error\b)",
    r"\b(yesterday'?s sales|vendas de ontem|vendas de hoje|today'?s sales)\b",
    # receivables asked without "my"/"minha" (found in manual testing: "Quanto vou receber hoje?")
    r"\b(quanto|o que) (eu )?(recebi|vou receber|tenho a receber|caiu|cai)\b",
    r"\b(recebi|vou receber|a receber)\b.*\b(hoje|ontem|amanha|semana|mes)\b",
    r"\b(what|how much) (did|will) i (receive|get paid|be paid)\b",
    r"\b(cuanto|que) (recibi|voy a recibir)\b",
]
UNSUPPORTED_TERMS = [
    r"\b(poem|poema|poesia|homework|li[cç][aã]o de casa|tarefa escolar|write (me )?(a |an )?(essay|story|code|song)|"
    r"escreva (um|uma) (poema|reda[cç][aã]o|hist[oó]ria|c[oó]digo|m[uú]sica)|recipe|receita de bolo|piada|joke)\b",
]
GENERAL_QUESTION = [
    r"^\s*(do i need|is it necessary|preciso (ter|de)|e necessario|how does|how do|como funciona|what is|what are|"
    r"o que e|can i|posso|how many|quantas|quantos|qual a diferenca|what'?s the difference|necesito|puedo|que es)\b",
    # "How do I activate Pix?" is general; "How do I activate MY machine?" needs the merchant's terminals
    # (product decision on case kb-07-pt: route to support).
    r"^\s*(como|how (do|can) i) (ativo|ativar|habilito|habilitar|configuro|configurar|activate|set up|enable)\b"
    r"(?!.*\b(minha|meu|minhas|meus|my)\b)",
]
VAGUE = [r"^\s*(help|ajuda|socorro|oi|ola|hi|hello|hey|ayuda|problema|it'?s not working|n[aã]o funciona)\s*[!?.]*\s*$"]


def heuristic_route(text: str) -> dict:
    norm = normalize(text)
    lang = detect_language(text)

    def any_match(patterns: list[str]) -> bool:
        return any(re.search(p, norm) for p in patterns)

    if any_match(VAGUE):
        question = {
            "pt": "Claro! Pode me contar um pouco mais? É sobre sua maquininha, suas vendas/recebimentos ou algum produto Getnet?",
            "en": "Sure! Could you tell me a bit more? Is it about your card machine, your sales/payouts or a Getnet product?",
            "es": "¡Claro! ¿Puedes contarme un poco más? ¿Es sobre tu terminal, tus ventas/pagos o algún producto Getnet?",
        }[lang]
        return {"intent": "knowledge", "confidence": 0.3, "language": lang,
                "reasoning": "Message too vague to route.", "clarifying_question": question}
    if any_match(ESCALATION_TERMS):
        return {"intent": "escalation", "confidence": 0.85, "language": lang,
                "reasoning": "Customer asks for a human or reports fraud/complaint.", "clarifying_question": None}
    if any_match(UNSUPPORTED_TERMS):
        return {"intent": "unsupported", "confidence": 0.8, "language": lang,
                "reasoning": "Request outside the assistant's scope.", "clarifying_question": None}
    if any_match(SUPPORT_TERMS) and not any_match(GENERAL_QUESTION):
        return {"intent": "support", "confidence": 0.8, "language": lang,
                "reasoning": "Needs the merchant's own data or equipment status.", "clarifying_question": None}
    return {"intent": "knowledge", "confidence": 0.7, "language": lang,
            "reasoning": "Public product or general-purpose question.", "clarifying_question": None}


# --------------------------------------------------------------------------- argument extraction
def extract_city(text: str) -> str:
    m = re.search(r"\b(?:in|em|en|para|for|de)\s+([A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+(?:[A-ZÀ-Ý][\wÀ-ÿ]+|de|do|da|dos))*)", text)
    return m.group(1).strip() if m else "São Paulo"


def extract_day_offset(text: str) -> int:
    n = normalize(text)
    if re.search(r"\b(tomorrow|amanha|manana)\b", n):
        return 1
    if re.search(r"\b(depois de amanha|day after tomorrow)\b", n):
        return 2
    return 0


CURRENCIES = {"euro": "EUR", "eur": "EUR", "dolar": "USD", "dollar": "USD", "usd": "USD", "libra": "GBP",
              "pound": "GBP", "gbp": "GBP", "peso argentino": "ARS", "iene": "JPY", "yen": "JPY", "yuan": "CNY", "bitcoin": "BTC"}


def extract_currency(text: str) -> str:
    n = normalize(text)
    for word, code in CURRENCIES.items():
        if word in n:
            return code
    return "USD"


def extract_error_codes(text: str) -> list[str]:
    codes = re.findall(r"\b(ADQ\s?\d-\d{2}|IDL\s?\d-\d{2})\b", text, re.IGNORECASE)
    codes += re.findall(r"(?:c[oó]digo|code|erro|error)\s*(\d{2})\b", text, re.IGNORECASE)
    return codes
