param(
    [string]$GroqApiKey = "",
    [int]$NBoth = 50,
    [int]$NCustom = 100
)

$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = "c:/Users/dongz/OneDrive/Project Code/LLM_Hallucination/llm-env/Scripts/python.exe"

if (-not (Test-Path $py)) {
    throw "Python not found at $py"
}

if ([string]::IsNullOrWhiteSpace($GroqApiKey)) {
    $GroqApiKey = $env:GROQ_API_KEY
}

if (-not $GroqApiKey -or -not $GroqApiKey.StartsWith("gsk_")) {
    throw "No valid Groq key found. Set GROQ_API_KEY or pass -GroqApiKey 'gsk_...'"
}

$env:GROQ_API_KEY = $GroqApiKey

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
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py @Args 2>&1 | Out-File -FilePath $LogPath -Encoding utf8
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevErrorAction

    if ($exitCode -ne 0) {
        throw "$StepName failed. Check $LogPath"
    }
}

# 0) Quick routing smoke check
Invoke-NativeLogged -StepName "Groq smoke" -LogPath "logs/groq_smoke.log" -Args @(
    "cured.py",
    "--api-mode", "groq",
    "--api-model", "llama-3.3-70b-versatile",
    "--question", "Does aspirin reduce cardiovascular risk in elderly patients with hypertension?"
)

# 1) TruthfulQA + MedHallu on main Groq models
Invoke-NativeLogged -StepName "Groq both benchmarks" -LogPath "logs/groq_both.log" -Args @(
    "cured.py",
    "--api-mode", "groq",
    "--api-model", "llama-3.3-70b-versatile,meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3-32b,openai/gpt-oss-120b",
    "--protocols", "greedy,cove,cured_api",
    "--benchmark", "both",
    "--n", "$NBoth",
    "--out", "results/results_groq_both.json"
)

# 2) MedQA custom benchmark
Invoke-NativeLogged -StepName "Groq MedQA custom" -LogPath "logs/groq_medqa.log" -Args @(
    "cured.py",
    "--api-mode", "groq",
    "--api-model", "llama-3.3-70b-versatile,meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3-32b",
    "--protocols", "greedy,cove,cured_api",
    "--benchmark", "custom",
    "--custom-csv", "benchmarks/medqa_usmle_n200.csv",
    "--scoring", "letter",
    "--max-new-tokens", "40",
    "--question-col", "question",
    "--answer-col", "answer",
    "--n", "$NCustom",
    "--out", "results/results_groq_medqa.json"
)

# 3) PubMedQA custom benchmark
Invoke-NativeLogged -StepName "Groq PubMedQA custom" -LogPath "logs/groq_pubmedqa.log" -Args @(
    "cured.py",
    "--api-mode", "groq",
    "--api-model", "llama-3.3-70b-versatile,meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3-32b",
    "--protocols", "greedy,cove,cured_api",
    "--benchmark", "custom",
    "--custom-csv", "benchmarks/pubmedqa_n200.csv",
    "--scoring", "yesno",
    "--max-new-tokens", "10",
    "--question-col", "question",
    "--answer-col", "answer",
    "--n", "$NCustom",
    "--out", "results/results_groq_pubmedqa.json"
)

Write-Output "Groq online run sequence completed."
Write-Output "Logs: logs/groq_smoke.log, logs/groq_both.log, logs/groq_medqa.log, logs/groq_pubmedqa.log"
