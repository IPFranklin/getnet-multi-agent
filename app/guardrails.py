"""Deterministic guardrails that run before and after the LLM agents.

Layer 1 (here, cheap and deterministic, < 1 ms):
  * input validation (size, empty)
  * sensitive-data detection & redaction (card PAN with Luhn check, CPF, CNPJ, CVV, passwords)
  * prompt-injection / jailbreak heuristics
  * unsafe-intent detection (fraud, card cloning, chargeback abuse, violence...)
Layer 2 (in the agents): system prompts restrict scope, tools are scoped to the
authenticated ``user_id`` (no IDOR), and the router can mark a request ``unsupported``.
Layer 3 (output): PII redaction on the final answer and empty/oversized answer protection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.observability import GUARDRAIL_EVENTS


def normalize(text: str) -> str:
    """Lowercase + strip accents, used for keyword matching in PT/EN/ES."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


# --------------------------------------------------------------------------- PII
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_CPF = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_CNPJ = re.compile(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)")
_CVV = re.compile(r"\b(?:cvv|cvc|c[oó]digo de seguran[cç]a)\b\D{0,10}\d{3,4}", re.IGNORECASE)
_PASSWORD = re.compile(
    r"\b(?:senha|password|pwd|pin)\b\s*(?:[:=]|\bis\b|\bé\b|\be\b)\s*\S*\d\S*", re.IGNORECASE
)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, categories_found)."""
    found: list[str] = []

    def _card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card_number")
            return f"[CARTÃO ****{digits[-4:]}]"
        return m.group(0)

    text = _CARD_CANDIDATE.sub(_card, text)
    if _CNPJ.search(text):
        found.append("cnpj")
        text = _CNPJ.sub("[CNPJ REDACTED]", text)
    if _CPF.search(text):
        found.append("cpf")
        text = _CPF.sub("[CPF REDACTED]", text)
    if _CVV.search(text):
        found.append("cvv")
        text = _CVV.sub("[CVV REDACTED]", text)
    if _PASSWORD.search(text):
        found.append("password")
        text = _PASSWORD.sub("[SENHA REDACTED]", text)
    return text, found


# --------------------------------------------------------------------------- injection / unsafe
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"ignore (todas )?(as )?instru[cç][oõ]es (anteriores|acima)",
    r"esque[cç]a (todas )?(as )?(suas )?instru[cç][oõ]es",
    r"(reveal|show|print|repeat) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"(mostre|revele|imprima) (o |seu )?(prompt|instru[cç][oõ]es) (do sistema|inicial)",
    r"you are now (?:dan|in developer mode|an? unrestricted)",
    r"developer mode|jailbreak|do anything now",
    r"act as (?:an? )?(?:admin|administrator|root|system)",
    r"(user_id|usu[aá]rio)\s*[:=]\s*\w+.*(instead|ao inv[eé]s|em vez)",
    r"</?(system|assistant|tool)>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_UNSAFE_PATTERNS = {
    "fraud": [
        r"clon(ar|e|ing) (de )?(um )?cart(ao|oes|ão|ões|ao de credito)|clone (a )?card",
        r"(card|cartao|cartão)s? (roubado|clonado|stolen)",
        r"(burlar|bypass|enganar|fool|evade|driblar) (o |a |the )?(sistema )?(antifraude|anti-?fraud|fraud detection|3ds)",
        r"(fake|falso|falsa) (chargeback|estorno)|(chargeback|estorno) (falso|fake)|chargeback fraud",
        r"lavar dinheiro|lavagem de dinheiro|launder (money)?",
        r"carding|cvv dump|bin attack|teste de cart(ao|oes) roubados",
    ],
    "violence": [r"\b(matar|kill|bomb|bomba|explosivo|arma de fogo)\b"],
    "self_harm": [r"\b(suic[ií]dio|me matar|kill myself|self[- ]harm)\b"],
}
_UNSAFE_RE = {cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in _UNSAFE_PATTERNS.items()}


@dataclass
class GuardResult:
    allowed: bool
    sanitized_text: str
    categories: list[str] = field(default_factory=list)
    reason: str | None = None
    response: str | None = None  # canned reply when blocked
    warnings: list[str] = field(default_factory=list)


_BLOCK_MESSAGES = {
    "prompt_injection": (
        "I can only help with questions about Getnet products, your account and payments. "
        "I can't change my instructions or reveal internal configuration. How can I help you today?"
    ),
    "fraud": (
        "I can't help with that. If you suspect fraud on your account or terminal, contact "
        "Getnet support immediately so we can protect your business."
    ),
    "violence": "I can't help with that request. I'm here for questions about Getnet products and your account.",
    "self_harm": (
        "I'm really sorry you're going through this. Please reach out to someone you trust or to "
        "CVV (Centro de Valorização da Vida) by calling 188 in Brazil, available 24h."
    ),
    "empty": "It looks like your message was empty. How can I help you?",
    "too_long": "Your message is too long. Please summarise your question in a few sentences.",
}


def check_input(text: str, max_chars: int = 2000) -> GuardResult:
    if not text or not text.strip():
        return _blocked(text, "empty")
    if len(text) > max_chars:
        return _blocked(text, "too_long")

    sanitized, pii = redact_pii(text)
    warnings = []
    for cat in pii:
        GUARDRAIL_EVENTS.labels(stage="input", category=cat, action="redacted").inc()
    if pii:
        warnings.append(
            "For your security, never share full card numbers, CVV, passwords or documents in chat. "
            "I removed that data from your message."
        )

    for rx in _INJECTION_RE:
        if rx.search(text):
            return _blocked(sanitized, "prompt_injection", pii)
    norm = normalize(text)
    for cat, rxs in _UNSAFE_RE.items():
        if any(rx.search(norm) or rx.search(text) for rx in rxs):
            return _blocked(sanitized, cat, pii)

    return GuardResult(allowed=True, sanitized_text=sanitized, categories=pii, warnings=warnings)


def _blocked(text: str, category: str, extra: list[str] | None = None) -> GuardResult:
    GUARDRAIL_EVENTS.labels(stage="input", category=category, action="blocked").inc()
    return GuardResult(
        allowed=False,
        sanitized_text=text,
        categories=[category, *(extra or [])],
        reason=category,
        response=_BLOCK_MESSAGES[category],
    )


_SYSTEM_LEAK = re.compile(r"(?i)(system prompt|<\s*/?system\s*>|you are the router agent)")


def check_output(text: str, max_chars: int = 6000) -> tuple[str, list[str]]:
    """Sanitise the final answer. Returns (text, categories_found)."""
    categories: list[str] = []
    if not text or not text.strip():
        categories.append("empty_output")
        text = (
            "Sorry, I couldn't produce an answer right now. A human specialist can help you "
            "through Getnet's support channels."
        )
    text, pii = redact_pii(text)
    categories += pii
    if _SYSTEM_LEAK.search(text):
        categories.append("system_leak")
        text = _SYSTEM_LEAK.sub("[redacted]", text)
    if len(text) > max_chars:
        categories.append("truncated")
        text = text[: max_chars - 3] + "..."
    for cat in categories:
        GUARDRAIL_EVENTS.labels(stage="output", category=cat, action="sanitized").inc()
    return text, categories
