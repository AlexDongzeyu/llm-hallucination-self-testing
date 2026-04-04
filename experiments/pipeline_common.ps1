$ErrorActionPreference = 'Stop'

function Initialize-PipelineContext {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot
    )

    $root = Split-Path -Parent $ScriptRoot
    $py = Join-Path $root 'llm-env/Scripts/python.exe'
    $logs = Join-Path $root 'results/logs'

    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    Set-Location $root

    return @{
        Root = $root
        Python = $py
        Logs = $logs
    }
}

function Invoke-PipelineStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string[]]$PyArgs,
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [string]$FailureHint = 'pipeline log'
    )

    Write-Host "[$(Get-Date -Format s)] START $Name"
    & $PythonPath @PyArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "Step failed: $Name (exit code=$code). See $FailureHint for details."
    }
    Write-Host "[$(Get-Date -Format s)] DONE  $Name"
}
