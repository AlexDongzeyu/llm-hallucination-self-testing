# ============================================================================
# run_all_local.ps1 — Windows PowerShell master run script
#
# Runs all CURED experiments sequentially on local GPU (RTX 5060 / CUDA).
# Step 1: Smoke test   (n=20,  ~5 min)
# Step 2: Phase 4 8B   (n=500, ~2 hrs)
# Step 3: ALTA ablation (n=200, ~45 min)
# Step 4: Greedy ablation (n=200, ~30 min)
# Step 5: FACTOR news   (n=200, ~20 min)
# Step 6: FACTOR wiki   (n=200, ~20 min)
# Step 7: Semantic entropy ablation (n=50 k=5, ~40 min)
# Step 8: Final statistics + R2 analysis
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_local.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_local.ps1 -SkipSmoke -SkipFactor
# ============================================================================

param(
    [switch]$SkipSmoke,
    [switch]$SkipFactor,
    [switch]$SkipSemantic,
    [string]$Model = "meta-llama/Llama-3.1-8B-Instruct",
    [string]$ResultsDir = "results/CANONICAL_v2"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$TS = (Get-Date -Format "yyyyMMdd_HHmmss")
$null = New-Item -ItemType Directory -Force -Path "logs", $ResultsDir, "benchmarks"

function Log { param($msg) Write-Host "[$([datetime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] $msg" }
function Die { param($msg) Log "ERROR: $msg"; exit 1 }

function Run-Step {
    param($Step, $Desc, $Args, $OutFile, $LogFile, $Optional=$false)
    Log "=== STEP $Step : $Desc ==="
    Log "  -> $OutFile"
    $cmd = "python -u cured.py $Args --out `"$OutFile`""
    Log "  CMD: $cmd"
    $proc = Start-Process -FilePath "python" -ArgumentList ("-u cured.py $Args --out `"$OutFile`"") `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
        -NoNewWindow -PassThru -Wait
    $merged = Get-Content $LogFile, "$LogFile.err" -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        if ($Optional) { Log "  WARN: Step $Step failed (optional). Check $LogFile" }
        else { Die "Step $Step failed (exit $($proc.ExitCode)). Check $LogFile" }
    } else {
        Log "  Done: $OutFile"
    }
}

# ── Verify thresholds ────────────────────────────────────────────────────────
$cfg = python -c "import json; d=json.load(open('configs/router_thresholds.json')); print(d['tau_kappa'], d['tau_ECR'], d['profile_mean_r2'])"
Log "Thresholds: $cfg"
$tau_k = [float]($cfg.Split()[0])
if ($tau_k -lt 0.5) { Die "tau_kappa=$tau_k still wrong (should be ~0.70)" }
Log "Thresholds verified OK."

# ── Step 1: Smoke test ───────────────────────────────────────────────────────
if (-not $SkipSmoke) {
    $SmokeOut = "logs/smoke_test_$TS.json"
    Log "=== STEP 1: Smoke test (Gate 2 verification, n=20) ==="
    $proc = Start-Process -FilePath "python" -ArgumentList (
        "-u cured.py --model `"$Model`" --load-in-4bit",
        "--protocols cured --router new --router-config configs/router_thresholds.json",
        "--benchmark truthfulqa --n 20 --save-per-question --out `"$SmokeOut`""
    ) -RedirectStandardOutput "logs/smoke_$TS.log" -RedirectStandardError "logs/smoke_$TS.err" `
        -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) { Die "Smoke test failed. Check logs/smoke_$TS.log" }
    python -c @"
import json, sys
d = json.load(open('$SmokeOut'))
def find_routing(obj, depth=0):
    if depth > 5: return None
    if isinstance(obj, dict):
        if 'routing' in obj: return obj['routing']
        for v in obj.values():
            r = find_routing(v, depth+1)
            if r: return r
    return None
routing = find_routing(d)
print('Routing distribution:', routing)
if routing and all(k == 'greedy_gate5' for k in routing):
    print('WARNING: All greedy_gate5 - Gate 2 not firing!')
    sys.exit(1)
print('Gate 2 firing correctly.')
"@
    Log "Smoke test passed."
} else { Log "Skipping smoke test." }

# ── Step 2: Phase 4 8B TruthfulQA ─────────────────────────────────────────
$Phase4Out = "$ResultsDir/main_cured_8b_truthfulqa_n500_v2.json"
Run-Step 2 "Phase 4 8B TruthfulQA (n=500, fixed router)" `
    "--model `"$Model`" --load-in-4bit --protocols cured --router new --router-config configs/router_thresholds.json --benchmark truthfulqa --n 500 --seed 42 --no-shuffle --scoring cosine --save-per-question --skip-iti" `
    $Phase4Out "logs/phase4_8b_$TS.log"

# ── Step 3: ALTA ablation (n=200, saves r2_q features) ────────────────────
$AltaAbl = "$ResultsDir/ablation_8b_alta_truthfulqa_n200.json"
Run-Step 3 "ALTA ablation 8B (n=200, --save-per-question)" `
    "--model `"$Model`" --load-in-4bit --model-params-b 8.0 --protocols alta --benchmark truthfulqa --n 200 --seed 42 --no-shuffle --scoring cosine --save-per-question --skip-iti" `
    $AltaAbl "logs/ablation_8b_alta_$TS.log"

# ── Step 4: Greedy ablation ─────────────────────────────────────────────────
$GreedyAbl = "$ResultsDir/ablation_8b_greedy_truthfulqa_n200.json"
Run-Step 4 "Greedy ablation 8B (n=200, --save-per-question)" `
    "--model `"$Model`" --load-in-4bit --model-params-b 8.0 --protocols greedy --benchmark truthfulqa --n 200 --seed 42 --no-shuffle --scoring cosine --save-per-question --skip-iti" `
    $GreedyAbl "logs/ablation_8b_greedy_$TS.log"

# ── Steps 5-6: FACTOR ─────────────────────────────────────────────────────
if (-not $SkipFactor) {
    if (-not (Test-Path "benchmarks/factor_news_n200.csv")) {
        Log "Downloading FACTOR data..."
        python scripts/prep_factor_benchmark.py
    } else { Log "FACTOR CSVs already present." }

    foreach ($subset in @("news", "wiki")) {
        $csv = "benchmarks/factor_${subset}_n200.csv"
        if (-not (Test-Path $csv)) { Log "WARN: $csv missing, skipping."; continue }
        $step = if ($subset -eq "news") { 5 } else { 6 }
        $factorOut = "$ResultsDir/results_8b_factor_${subset}_n200.json"
        Run-Step $step "FACTOR $subset (n=200, letter scoring)" `
            "--model `"$Model`" --load-in-4bit --skip-iti --protocols greedy,alta,cured --router new --router-config configs/router_thresholds.json --benchmark custom --custom-csv `"$csv`" --question-col question --answer-col answer --n 200 --seed 42 --no-shuffle --scoring letter --max-new-tokens 5 --save-per-question" `
            $factorOut "logs/8b_factor_${subset}_$TS.log" -Optional
    }
} else { Log "Skipping FACTOR steps." }

# ── Step 7: Semantic entropy ablation ────────────────────────────────────
if (-not $SkipSemantic) {
    $SeOut = "$ResultsDir/semantic_entropy_gate_comparison.json"
    Log "=== STEP 7: Semantic entropy ablation (MedHallu, n=50, k=5) ==="
    $proc = Start-Process -FilePath "python" -ArgumentList (
        "-u experiments/run_semantic_entropy_ablation.py",
        "--model `"$Model`" --load-in-4bit --benchmark medhallu --n 50 --k 5 --seed 42 --out `"$SeOut`""
    ) -RedirectStandardOutput "logs/semantic_entropy_$TS.log" -RedirectStandardError "logs/semantic_entropy_$TS.err" `
        -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) { Log "WARN: Semantic entropy ablation failed. Check logs/semantic_entropy_$TS.log" }
    else { Log "Semantic entropy ablation done -> $SeOut" }
} else { Log "Skipping semantic entropy." }

# ── Step 8: Final statistics ──────────────────────────────────────────────
Log "=== STEP 8: Computing final statistics + R2 stratified analysis ==="
$proc = Start-Process -FilePath "python" -ArgumentList (
    "compute_final_stats.py --results-dir `"$ResultsDir`" --output `"$ResultsDir/statistics_table.json`""
) -RedirectStandardOutput "logs/final_stats_$TS.log" -RedirectStandardError "logs/final_stats_$TS.err" `
    -NoNewWindow -PassThru -Wait
Get-Content "logs/final_stats_$TS.log", "logs/final_stats_$TS.err" -ErrorAction SilentlyContinue | Write-Host
Log "Statistics -> $ResultsDir/statistics_table.json"

Log ""
Log "=== ALL RUNS COMPLETE ==="
Log "Phase 4:       $Phase4Out"
Log "ALTA ablation: $AltaAbl"
Log "Greedy abl.:   $GreedyAbl"
Log "FACTOR news:   $ResultsDir/results_8b_factor_news_n200.json"
Log "FACTOR wiki:   $ResultsDir/results_8b_factor_wiki_n200.json"
Log "Stats:         $ResultsDir/statistics_table.json"
Log "R2 stratified: $ResultsDir/r2_stratified_analysis.json"
