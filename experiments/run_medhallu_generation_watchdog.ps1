param(
    [int]$N = 50,
    [double]$Threshold = 0.65,
    [string]$Out = "results/medhallu_generation_results.json",
    [int]$RetryDelaySec = 10
)

$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$python = "c:/Users/dongz/OneDrive/Project Code/LLM_Hallucination/llm-env/Scripts/python.exe"
$logDir = "results/logs/manual_runs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$requiredLabels = @("greedy", "cove", "gadr2_cured", "cove_rag", "delta_dola")

function Test-GenerationComplete {
    param([string]$Path, [string[]]$Labels)

    if (-not (Test-Path $Path)) {
        return $false
    }

    try {
        $json = Get-Content -Path $Path -Raw | ConvertFrom-Json
        $done = @{}
        if ($null -ne $json.results) {
            foreach ($r in $json.results) {
                if ($null -ne $r.label -and "$($r.label)" -ne "") {
                    $done["$($r.label)"] = $true
                }
            }
        }

        foreach ($label in $Labels) {
            if (-not $done.ContainsKey($label)) {
                return $false
            }
        }

        return $true
    }
    catch {
        return $false
    }
}

$attempt = 0
while (-not (Test-GenerationComplete -Path $Out -Labels $requiredLabels)) {
    $attempt += 1
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $outLog = "$logDir/medhallu_generation_attempt_${attempt}_${ts}.out.log"
    $errLog = "$logDir/medhallu_generation_attempt_${attempt}_${ts}.err.log"

    Write-Output "[Attempt $attempt] Starting run_medhallu_generation.py"
    Write-Output "  OUT: $outLog"
    Write-Output "  ERR: $errLog"

    & $python -u "experiments/run_medhallu_generation.py" `
        --n $N `
        --threshold $Threshold `
        --out $Out `
        --resume `
        1>> $outLog 2>> $errLog

    $exitCode = $LASTEXITCODE

    if (Test-GenerationComplete -Path $Out -Labels $requiredLabels) {
        break
    }

    Write-Output "[Attempt $attempt] Incomplete run (exit code $exitCode). Retrying in $RetryDelaySec s..."
    Start-Sleep -Seconds $RetryDelaySec
}

Write-Output "Generation run complete: $Out"
