$ErrorActionPreference = 'Stop'

# End-to-end orchestrator for dataset build, router training, and final evaluations.

. (Join-Path $PSScriptRoot 'pipeline_common.ps1')
$ctx = Initialize-PipelineContext -ScriptRoot $PSScriptRoot
$root = $ctx.Root
$py = $ctx.Python
$logs = $ctx.Logs

$routingCsv = Join-Path $root 'results/routing_dataset.csv'
$routerJoblib = Join-Path $root 'results/router_model.joblib'

if (Test-Path $routingCsv) {
    Remove-Item $routingCsv -Force
}

Invoke-PipelineStep -Name 'build_routing_dataset' -PythonPath $py -PyArgs @('-u', 'experiments/build_routing_dataset.py') -LogPath (Join-Path $logs 'build_routing_dataset.log') -FailureHint 'orchestrator log'

if (-not (Test-Path $routingCsv)) {
    throw "Missing routing dataset after Step 1: $routingCsv"
}

$lineCount = (Get-Content $routingCsv | Measure-Object -Line).Lines
if ($lineCount -lt 2) {
    throw "Routing dataset is empty or header-only (lines=$lineCount): $routingCsv"
}

Invoke-PipelineStep -Name 'learn_router' -PythonPath $py -PyArgs @('src/learn_router.py') -LogPath (Join-Path $logs 'learn_router.log') -FailureHint 'orchestrator log'

if (-not (Test-Path $routerJoblib)) {
    throw "Missing trained router after Step 2: $routerJoblib"
}

Invoke-PipelineStep -Name 'eval_instruct' -PythonPath $py -PyArgs @('-u', 'experiments/eval_instruct.py') -LogPath (Join-Path $logs 'eval_instruct.log') -FailureHint 'orchestrator log'

Invoke-PipelineStep -Name 'eval_medhallu' -PythonPath $py -PyArgs @('-u', 'experiments/eval_medhallu.py') -LogPath (Join-Path $logs 'eval_medhallu.log') -FailureHint 'orchestrator log'

$donePath = Join-Path $logs 'pipeline_all_steps.done'
"COMPLETED $(Get-Date -Format s)" | Out-File -FilePath $donePath -Encoding ascii -Force
Write-Host "All steps completed. Marker: $donePath"
