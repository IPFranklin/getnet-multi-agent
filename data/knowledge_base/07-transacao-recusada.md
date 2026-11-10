---
title: Transação recusada ou negada na maquininha — causas e o que fazer
url: kb://transacao-recusada
tags: [suporte, recusa, transacao-negada, codigos-de-resposta, troubleshooting]
language: pt
---

# Minha maquininha mostra erro de transação recusada

Quando uma venda é recusada, a maquininha exibe uma mensagem e/ou um código de resposta. Na maioria dos casos a recusa vem do **banco emissor do cartão do cliente** (limite, saldo, bloqueio, suspeita de fraude), e não de um defeito da maquininha.

## Primeiros passos

1. Leia a mensagem/código exibido e confira o comprovante de não aprovação (ou o relatório de vendas no Aplicativo Getnet Brasil).
2. Se a recusa for do emissor (ex.: saldo/limite insuficiente, não autorizada), peça ao cliente outro cartão, outra modalidade (débito/crédito) ou pagamento via **Pix**.
3. Se várias transações de **cartões diferentes** forem recusadas em sequência, o problema pode ser da maquininha ou da conexão: verifique a internet, reinicie a maquininha e tente novamente.
4. Não insista várias vezes com o mesmo cartão em caso de suspeita de fraude — oriente o cliente a falar com o banco.
5. Se o erro persistir para todos os cartões, contate o suporte Getnet pelo app ou pela Central de Atendimento.

## Códigos de resposta mais comuns (padrão de mercado)

- **05** — Não autorizada pelo emissor (recusa genérica): cliente deve contatar o banco ou usar outro meio.
- **51** — Saldo/limite insuficiente: outro cartão, menos parcelas, dividir o pagamento ou Pix.
- **54** — Cartão vencido.
- **55** — Senha incorreta: digitar novamente com atenção (após 3 erros o cartão é bloqueado pelo emissor).
- **57** — Transação não permitida para o cartão: tentar outra função ou outro cartão.
- **59** — Suspeita de fraude: não repetir; cliente deve contatar o banco.
- **91** — Emissor indisponível: aguardar 1–2 minutos e tentar de novo.
- **96** — Falha de sistema: reiniciar a maquininha; persistindo, contatar o suporte.
