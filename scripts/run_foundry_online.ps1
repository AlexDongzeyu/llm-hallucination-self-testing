param(
    [string]$FoundryApiKey = "",
    [string]$FoundryBaseUrl = "https://llm-hallucination-resource.openai.azure.com/openai/v1",
    [string]$FoundryModel = "gpt-4o-mini",
    [int]$NBoth = 50,
    [int]$NCustom = 100
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = "c:/Users/dongz/OneDrive/Project Code/LLM_Hallucination/llm-env/Scripts/python.exe"
if (-not (Test-Path $py)) {
    throw "Python not found at $py"
}

if ([string]::IsNullOrWhiteSpace($FoundryApiKey)) {
    $FoundryApiKey = $env:FOUNDRY_API_KEY
}

if ([string]::IsNullOrWhiteSpace($FoundryApiKey)) {
    throw "No Foundry key found. Pass -FoundryApiKey or set FOUNDRY_API_KEY."
}

$env:FOUNDRY_API_KEY = $FoundryApiKey
$env:FOUNDRY_BASE_URL = $FoundryBaseUrl
$env:FOUNDRY_MODEL = $FoundryModel

Write-Output "Foundry smoke test..."
& $py -u cured.py --api-mode foundry --api-model $FoundryModel --question "Answer in one word: ok"

Write-Output "Foundry both benchmark..."
& $py -u cured.py --api-mode foundry --api-model $FoundryModel --protocols greedy,cove,cured_api --benchmark both --n $NBoth --max-new-tokens 16 --out results/results_foundry_both.json

Write-Output "Foundry MedQA benchmark..."
& $py -u cured.py --api-mode foundry --api-model $FoundryModel --protocols greedy,cove,cured_api --benchmark custom --custom-csv benchmarks/medqa_usmle_n200.csv --question-col question --answer-col answer --n $NCustom --max-new-tokens 16 --out results/results_foundry_medqa.json

Write-Output "Foundry PubMedQA benchmark..."
& $py -u cured.py --api-mode foundry --api-model $FoundryModel --protocols greedy,cove,cured_api --benchmark custom --custom-csv benchmarks/pubmedqa_n200.csv --question-col question --answer-col answer --n $NCustom --max-new-tokens 16 --out results/results_foundry_pubmedqa.json

Write-Output "Done."
