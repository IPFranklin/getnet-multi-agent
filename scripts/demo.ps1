# Envia os 10 cenarios do desafio para a API rodando localmente.
# Equivalente Windows/PowerShell do scripts/demo.sh.
# Uso:  .\scripts\demo.ps1                (padrao: http://localhost:8000, cliente1988)
#       .\scripts\demo.ps1 -UserId cliente2024
#       .\scripts\demo.ps1 -SharedSession   (todas as perguntas na MESMA conversa - reproduz o efeito
#                                             da memoria compartilhada entre agentes)
# Por padrao cada pergunta vai numa conversa (session_id) propria, como no harness de evals.
param(
    [string]$Url = "http://localhost:8000",
    [string]$UserId = "cliente1988",
    [switch]$SharedSession
)
$runId = Get-Date -Format "HHmmss"
$n = 0

$questions = @(
    "What's the difference between the Get Clássica and the Get Smart?",
    "What's the weather forecast in Porto Alegre tomorrow?",
    "When will the money from yesterday's sales be deposited?",
    "Do I need a bank account to receive my sales via Pix?",
    "My card machine won't connect to the internet, what should I do?",
    "How does receivables advance (antecipação) work with Getnet?",
    "What's the euro exchange rate today?",
    "My card machine is showing a transaction decline error.",
    "How many installments can I split a sale into with the crediário?",
    "Can I sell through WhatsApp using the Payment Link?"
)

foreach ($q in $questions) {
    $n++
    $payload = @{ message = $q; user_id = $UserId }
    if (-not $SharedSession) { $payload.session_id = "demo-$runId-$n" }
    Write-Host ("-" * 78)
    Write-Host "PERGUNTA: $q" -ForegroundColor Cyan
    $body = [Text.Encoding]::UTF8.GetBytes(($payload | ConvertTo-Json -Compress))
    try {
        $resp = Invoke-WebRequest -Uri "$Url/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body $body -UseBasicParsing
    } catch {
        Write-Host "ERRO ao chamar a API: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "A API esta rodando? (uvicorn app.main:app --port 8000)" -ForegroundColor Red
        break
    }
    # Decodifica como UTF-8 para manter os acentos corretos no Windows PowerShell 5.1
    $r = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray()) | ConvertFrom-Json
    $tools = ($r.tools_used | ForEach-Object { $_.tool }) -join ", "
    $nodes = ($r.trace | ForEach-Object { $_.node }) -join " -> "
    Write-Host "AGENTE: $($r.agent)   FERRAMENTAS: [$tools]   LATENCIA: $($r.latency_ms) ms" -ForegroundColor Yellow
    Write-Host "CAMINHO: $nodes" -ForegroundColor DarkYellow
    Write-Host ""
    Write-Host $r.response
    if ($r.sources) {
        Write-Host ""
        Write-Host ("FONTES: " + (($r.sources | ForEach-Object { $_.url }) -join "  ")) -ForegroundColor DarkGray
    }
    if ($r.handoff) {
        Write-Host ("HANDOFF: " + ($r.handoff | ConvertTo-Json -Compress)) -ForegroundColor Magenta
    }
}
Write-Host ("-" * 78)
