"""Diagnostico da chave da API da Anthropic (nao imprime a chave inteira).

Uso (no terminal com .venv, na pasta do projeto):
    python scripts/check_key.py

Le a chave do arquivo .env (via app.config) e testa uma chamada minima de duas formas:
  1) cabecalho x-api-key  (o jeito que o projeto/LangChain usa)
  2) cabecalho Authorization: Bearer
"""

from __future__ import annotations

import anthropic

from app.config import get_settings

s = get_settings()
key = (s.anthropic_api_key or "").strip()

print(f"LLM_PROVIDER lido do .env: {s.llm_provider}")
if not key:
    print("ERRO: ANTHROPIC_API_KEY vazia no .env")
    raise SystemExit(1)
print(f"Chave lida: comeca com '{key[:11]}...' e tem {len(key)} caracteres")
if key != (s.anthropic_api_key or ""):
    print("AVISO: a chave tinha espacos/quebras de linha nas pontas")

model = s.resolved_router_model()
print(f"Modelo de teste: {model}\n")


def tentar(nome: str, client: anthropic.Anthropic) -> None:
    try:
        msg = client.messages.create(model=model, max_tokens=20,
                                     messages=[{"role": "user", "content": "Responda apenas: oi"}])
        print(f"[OK]   {nome}: resposta = {msg.content[0].text!r}")
    except anthropic.APIStatusError as e:
        print(f"[FALHA] {nome}: HTTP {e.status_code} - {e.message}")
    except Exception as e:  # noqa: BLE001
        print(f"[FALHA] {nome}: {type(e).__name__}: {e}")


tentar("x-api-key", anthropic.Anthropic(api_key=key))
tentar("Authorization: Bearer", anthropic.Anthropic(auth_token=key))
