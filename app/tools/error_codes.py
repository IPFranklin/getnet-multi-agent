"""Catalogue of terminal and authorization response codes with merchant guidance.

Response codes follow the market-standard (ABECS/ISO 8583-style) authorization codes used by
Brazilian acquirers; ADQ/IDL codes come from the Getnet terminal manuals. Treat as an
illustrative catalogue — production would read it from the acquirer's code registry.
"""

from __future__ import annotations

ERROR_CODES: dict[str, dict] = {
    "00": {"meaning": "Approved", "who": "-", "action": "No action needed."},
    "05": {
        "meaning": "Not authorized by the card issuer (generic decline)", "who": "issuer",
        "action": "The decline comes from the customer's bank, not from the terminal. Ask the customer to "
                  "contact their bank or use another card/payment method (e.g. Pix).", "retry": False,
    },
    "51": {
        "meaning": "Insufficient funds / credit limit", "who": "issuer",
        "action": "Ask the customer to use another card, pay in fewer installments, split the payment or pay via Pix.",
        "retry": False,
    },
    "54": {"meaning": "Expired card", "who": "issuer", "action": "Ask for another card.", "retry": False},
    "55": {
        "meaning": "Wrong PIN", "who": "cardholder",
        "action": "Ask the customer to type the PIN again carefully. After 3 wrong attempts the card is blocked by the issuer.",
        "retry": True,
    },
    "57": {
        "meaning": "Transaction not permitted for this card", "who": "issuer",
        "action": "The card does not allow this type of purchase (e.g. credit on a debit-only card, or online/installments "
                  "disabled). Try another function (debit/credit) or another card.", "retry": True,
    },
    "59": {
        "meaning": "Suspected fraud", "who": "issuer/anti-fraud",
        "action": "Do not retry repeatedly. Ask the customer to contact their bank. If you see several of these in a row "
                  "with different cards, contact Getnet support.", "retry": False,
    },
    "61": {"meaning": "Exceeds withdrawal/amount limit", "who": "issuer", "action": "Try a lower amount or another card.", "retry": False},
    "75": {"meaning": "PIN tries exceeded", "who": "issuer", "action": "Card blocked for PIN; the customer must contact the bank.", "retry": False},
    "91": {
        "meaning": "Issuer unavailable / timeout", "who": "network",
        "action": "Temporary. Wait 1-2 minutes and try again, or use another payment method.", "retry": True,
    },
    "96": {
        "meaning": "System malfunction", "who": "network/acquirer",
        "action": "Restart the terminal and try again. If it persists for several cards, contact Getnet support.", "retry": True,
    },
    "ADQ 1-06": {
        "meaning": "Terminal not configured", "who": "terminal",
        "action": "Activate the terminal with the token from the Getnet Brasil app (Serviços > Token de Ativação).",
    },
    "ADQ 1-07": {
        "meaning": "Terminal not initialized", "who": "terminal",
        "action": "Check the connection and run the terminal initialization; if it fails, request support in the Getnet Brasil app.",
    },
    "IDL 1-08": {
        "meaning": "Integration error with card brands", "who": "acquirer",
        "action": "Restart the terminal and retry; if it persists, contact Getnet support.",
    },
    "NO_NETWORK": {
        "meaning": "Terminal without internet connection", "who": "terminal/connectivity",
        "action": "Check Wi-Fi/router, chip signal, enable the SIM fallback, restart the terminal.",
    },
}


def lookup(code: str) -> dict | None:
    key = code.strip().upper().replace("-", "-")
    if key in ERROR_CODES:
        return ERROR_CODES[key]
    digits = "".join(ch for ch in key if ch.isdigit())
    if len(digits) == 2 and digits in ERROR_CODES:
        return ERROR_CODES[digits]
    for k, v in ERROR_CODES.items():
        if k.replace(" ", "") == key.replace(" ", ""):
            return v
    return None
