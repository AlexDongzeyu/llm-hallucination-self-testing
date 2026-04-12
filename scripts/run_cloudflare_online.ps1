param(
    [string]$CloudflareApiToken = "",
    [string]$CloudflareAccountId = "",
    [string]$ApiModels = "@cf/meta/llama-3.1-8b-instruct",
    [int]$NBoth = 50,
    [int]$NCustom = 100
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = "c:/Users/dongz/OneDrive/Project Code/LLM_Hallucination/llm-env/Scripts/python.exe"
if (-not (Test-Path $py)) {
    throw "Python not found at $py"
}

if ([string]::IsNullOrWhiteSpace($CloudflareApiToken)) {
    $CloudflareApiToken = $env:CLOUDFLARE_API_TOKEN
}
if ([string]::IsNullOrWhiteSpace($CloudflareAccountId)) {
    $CloudflareAccountId = $env:CLOUDFLARE_ACCOUNT_ID
}

if ([string]::IsNullOrWhiteSpace($CloudflareApiToken)) {
    throw "No Cloudflare token found. Set CLOUDFLARE_API_TOKEN or pass -CloudflareApiToken."
}
if ([string]::IsNullOrWhiteSpace($CloudflareAccountId)) {
    throw "No Cloudflare account id found. Set CLOUDFLARE_ACCOUNT_ID or pass -CloudflareAccountId."
}

$env:CLOUDFLARE_API_TOKEN = $CloudflareApiToken
$env:CLOUDFLARE_ACCOUNT_ID = $CloudflareAccountId

function Invoke-NativeLogged {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StepName,
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Write-Output "==> $StepName"
    & $py @Args *> $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed. Check $LogPath"
    }
}

$firstModel = ($ApiModels -split ',')[0].Trim()

Invoke-NativeLogged -StepName "Cloudflare smoke" -LogPath "logs/cloudflare_smoke.log" -Args @(
    "cured.py",
    "--api-mode", "cloudflare",
    "--api-model", $firstModel,
    "--question", "Answer in one word: ok"
)

Invoke-NativeLogged -StepName "Cloudflare both benchmarks" -LogPath "logs/cloudflare_both.log" -Args @(
    "cured.py",
    "--api-mode", "cloudflare",
    "--api-model", $ApiModels,
    "--protocols", "greedy,cove,cured_api",
    "--benchmark", "both",
    "--n", "$NBoth",
    "--out", "results/results_cloudflare_both.json"
)

Invoke-NativeLogged -StepName "Cloudflare MedQA custom" -LogPath "logs/cloudflare_medqa.log" -Args @(
    "cured.py",
    "--api-mode", "cloudflare",
    "--api-model", $ApiModels,
    "--protocols", "greedy,cove,cured_api",
    "--benchmark", "custom",
    "--custom-csv", "benchmarks/medqa_usmle_n200.csv",
    "--question-col", "question",
    "--answer-col", "answer",
    "--n", "$NCustom",
    "--out", "results/results_cloudflare_medqa.json"
)

Invoke-NativeLogged -StepName "Cloudflare PubMedQA custom" -LogPath "logs/cloudflare_pubmedqa.log" -Args @(
    "cured.py",
    "--api-mode", "cloudflare",
    "--api-model", $ApiModels,
    "--protocols", "greedy,cove,cured_api",
    "--benchmark", "custom",
    "--custom-csv", "benchmarks/pubmedqa_n200.csv",
    "--question-col", "question",
    "--answer-col", "answer",
    "--n", "$NCustom",
    "--out", "results/results_cloudflare_pubmedqa.json"
)

Write-Output "Cloudflare online run sequence completed."
Write-Output "Logs: logs/cloudflare_smoke.log, logs/cloudflare_both.log, logs/cloudflare_medqa.log, logs/cloudflare_pubmedqa.log"
