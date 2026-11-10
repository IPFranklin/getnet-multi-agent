#!/usr/bin/env bash
# Sends the challenge example scenarios to the running API and prints agent + answer.
set -euo pipefail
URL="${URL:-http://localhost:8000}"
USER_ID="${USER_ID:-cliente1988}"
questions=(
  "What's the difference between the Get Clássica and the Get Smart?"
  "What's the weather forecast in Porto Alegre tomorrow?"
  "When will the money from yesterday's sales be deposited?"
  "Do I need a bank account to receive my sales via Pix?"
  "My card machine won't connect to the internet, what should I do?"
  "How does receivables advance (antecipação) work with Getnet?"
  "What's the euro exchange rate today?"
  "My card machine is showing a transaction decline error."
  "How many installments can I split a sale into with the crediário?"
  "Can I sell through WhatsApp using the Payment Link?"
)
for q in "${questions[@]}"; do
  echo "──────────────────────────────────────────────────────────────"
  echo "❓ $q"
  curl -s -X POST "$URL/chat" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"message": sys.argv[1], "user_id": sys.argv[2]}))' "$q" "$USER_ID")" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(f"🤖 agent={r[\"agent\"]} tools={[t[\"tool\"] for t in r[\"tools_used\"]]} latency={r[\"latency_ms\"]}ms"); print(r["response"]); print("📚", [s["url"] for s in r["sources"]])'
done
