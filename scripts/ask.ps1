# Envia UMA pergunta para a API rodando localmente e mostra o resultado de forma legivel.
# Uso:  .\scripts\ask.ps1 "sua pergunta"
#       .\scripts\ask.ps1 "sua pergunta" -UserId cliente2024
#       .\scripts\ask.ps1 "sua pergunta" -Raw      (mostra o JSON completo da resposta)
#       .\scripts\ask.ps1 "sua pergunta" -SessionId conversa1   (continua uma conversa especifica)
# Sem -SessionId a pergunta entra na conversa "default" do lojista (com o historico das anteriores).
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Message,
    [string]$UserId = "cliente1988",
    [string]$Url = "http://localhost:8000",
    [string]$SessionId,
    [switch]$Raw
)

$payload = @{ message = $Message; user_id = $UserId }
if ($SessionId) { $payload.session_id = $SessionId }
$body = [Text.Encoding]::UTF8.GetBytes(($payload | ConvertTo-Json -Compress))
try {
    $resp = Invoke-WebRequest -Uri "$Url/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body $body -UseBasicParsing
} catch {
    Write-Host "ERRO ao chamar a API: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "A API esta rodando? (uvicorn app.main:app --port 8000)" -ForegroundColor Red
    return
}
$json = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray())
if ($Raw) { $json | ConvertFrom-Json | ConvertTo-Json -Depth 10; return }

$r = $json | ConvertFrom-Json
$tools = ($r.tools_used | ForEach-Object { $_.tool }) -join ", "
$nodes = ($r.trace | ForEach-Object { $_.node }) -join " -> "
Write-Host ("-" * 78)
Write-Host "PERGUNTA ($UserId): $Message" -ForegroundColor Cyan
Write-Host "AGENTE: $($r.agent)   INTENCAO: $($r.route.intent) (confianca $($r.route.confidence), idioma $($r.route.language))" -ForegroundColor Yellow
Write-Host "FERRAMENTAS: [$tools]   LATENCIA: $($r.latency_ms) ms" -ForegroundColor Yellow
Write-Host "CAMINHO: $nodes" -ForegroundColor DarkYellow
$cats = @($r.guardrails.categories) + @($r.guardrails.output_categories) | Where-Object { $_ }
if ($cats) { Write-Host ("GUARDRAILS: permitido=$($r.guardrails.allowed)  categorias=[" + ($cats -join ", ") + "]") -ForegroundColor Red }
Write-Host ""
Write-Host $r.response
if ($r.sources) {
    Write-Host ""
    Write-Host ("FONTES: " + (($r.sources | ForEach-Object { $_.url }) -join "  ")) -ForegroundColor DarkGray
}
if ($r.handoff) { Write-Host ""; Write-Host ("HANDOFF: " + ($r.handoff | ConvertTo-Json -Compress)) -ForegroundColor Magenta }
if ($r.tickets) { Write-Host ""; Write-Host ("TICKETS: " + ($r.tickets | ConvertTo-Json -Compress)) -ForegroundColor Magenta }
Write-Host ("-" * 78)
