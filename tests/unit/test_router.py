import pytest
from langchain_core.runnables import RunnableLambda

from app.agents.heuristics import detect_language, extract_city, extract_currency, extract_day_offset, heuristic_route
from app.agents.router import RouterAgent

EXAMPLES = [
    ("What's the difference between the Get Clássica and the Get Smart?", "knowledge"),
    ("What's the weather forecast in Porto Alegre tomorrow?", "knowledge"),
    ("When will the money from yesterday's sales be deposited?", "support"),
    ("Do I need a bank account to receive my sales via Pix?", "knowledge"),
    ("My card machine won't connect to the internet, what should I do?", "support"),
    ("How does receivables advance (antecipação) work with Getnet?", "knowledge"),
    ("What's the euro exchange rate today?", "knowledge"),
    ("My card machine is showing a transaction decline error.", "support"),
    ("How many installments can I split a sale into with the crediário?", "knowledge"),
    ("Can I sell through WhatsApp using the Payment Link?", "knowledge"),
    ("Quero falar com um atendente", "escalation"),
    ("Tem uma venda que eu não reconheço, acho que é fraude", "escalation"),
    ("Escreva um poema sobre gatos", "unsupported"),
    # Regressions found in manual testing (Portuguese receivables questions without "minha"/"my")
    ("Quando cai o dinheiro das vendas de ontem?", "support"),
    ("Quanto vou receber hoje?", "support"),
    ("Quanto eu recebi ontem?", "support"),
    ("What did I receive yesterday?", "support"),
    ("Como funciona a antecipação de recebíveis?", "knowledge"),
]


@pytest.mark.parametrize("message,intent", EXAMPLES)
def test_heuristic_router(message, intent):
    assert heuristic_route(message)["intent"] == intent


def test_vague_message_asks_for_clarification():
    r = heuristic_route("help")
    assert r["confidence"] < 0.45 and r["clarifying_question"]


@pytest.mark.parametrize("text,lang", [("Quando vou receber minhas vendas?", "pt"), ("When will I get paid?", "en"),
                                       ("¿Cuándo recibo el dinero de mis ventas?", "es"),
                                       # regressions: used to default to "pt"
                                       ("Write a poem about cats", "en"), ("Who won the 2022 World Cup?", "en"),
                                       ("Qual a diferença entre a Get Clássica e a Get Smart?", "pt")])
def test_language_detection(text, lang):
    assert detect_language(text) == lang


def test_argument_extraction():
    assert extract_city("What's the weather forecast in Porto Alegre tomorrow?") == "Porto Alegre"
    assert extract_day_offset("previsão para amanhã") == 1
    assert extract_currency("cotação do euro hoje") == "EUR"


async def test_router_falls_back_to_heuristics_when_llm_fails(container):
    router = RouterAgent(container)

    async def boom(_):
        raise TimeoutError("llm down")

    router._structured = RunnableLambda(boom)
    route, method = await router.classify("My card machine is showing a transaction decline error.", [])
    assert method == "heuristic_fallback" and route["intent"] == "support"


async def test_low_confidence_produces_clarifying_question(container):
    router = RouterAgent(container)
    update = await router({"message": "help", "history": []})
    assert update["route"]["intent_final"] == "clarify" and update["response"]
