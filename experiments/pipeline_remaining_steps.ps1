$ErrorActionPreference = 'Stop'

# Resume orchestrator that assumes routing data already exists.

. (Join-Path $PSScriptRoot 'pipeline_common.ps1')
$ctx = Initialize-PipelineContext -ScriptRoot $PSScriptRoot
$root = $ctx.Root
$py = $ctx.Python
$logs = $ctx.Logs

$routerJoblib = Join-Path $root 'results/router_model.joblib'
$routingCsv = Join-Path $root 'results/routing_dataset.csv'
if (-not (Test-Path $routingCsv)) {
    throw "Missing routing dataset: $routingCsv"
}

Invoke-PipelineStep -Name 'learn_router' -PythonPath $py -PyArgs @('src/learn_router.py') -LogPath (Join-Path $logs 'learn_router.log') -FailureHint 'resume log'

if (-not (Test-Path $routerJoblib)) {
    throw "Missing trained router after Step 2: $routerJoblib"
}

Invoke-PipelineStep -Name 'eval_instruct' -PythonPath $py -PyArgs @('-u', 'experiments/eval_instruct.py') -LogPath (Join-Path $logs 'eval_instruct.log') -FailureHint 'resume log'
Invoke-PipelineStep -Name 'eval_medhallu' -PythonPath $py -PyArgs @('-u', 'experiments/eval_medhallu.py') -LogPath (Join-Path $logs 'eval_medhallu.log') -FailureHint 'resume log'

$donePath = Join-Path $logs 'pipeline_remaining_steps.done'
"COMPLETED $(Get-Date -Format s)" | Out-File -FilePath $donePath -Encoding ascii -Force
Write-Host "All remaining steps completed. Marker: $donePath"
