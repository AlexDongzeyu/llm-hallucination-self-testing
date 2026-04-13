@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
pushd "%ROOT%"

if "%~1"=="" (
  echo Usage: run_openrouter_job.cmd ^<both^|medqa^|pubmedqa^|medhallu^>
  exit /b 2
)

if exist "logs\runtime_api_env.txt" (
  for /f "usebackq tokens=1,* delims==" %%A in ("logs\runtime_api_env.txt") do (
    set "%%A=%%B"
  )
)

if "%OPENROUTER_API_KEY%"=="" (
  echo OPENROUTER_API_KEY is not set. Configure it in environment first.
  exit /b 3
)

set "JOB=%~1"
set "MODEL=meta-llama/llama-3.1-8b-instruct"
set "COMMON=--api-mode openrouter --api-model %MODEL% --protocols greedy,cove,cured_api"

if /i "%JOB%"=="both" (
  set "OUT=results\results_openrouter_both.json"
  set "LOG=logs\openrouter_both_full.log"
  set "EXITFILE=logs\openrouter_both_full.exit.txt"
  set "ARGS=--benchmark both --n 50 --scoring cosine --max-new-tokens 16"
) else if /i "%JOB%"=="medqa" (
  set "OUT=results\results_openrouter_medqa_v2.json"
  set "LOG=logs\openrouter_medqa_v2.log"
  set "EXITFILE=logs\openrouter_medqa_v2.exit.txt"
  set "ARGS=--benchmark custom --custom-csv benchmarks\medqa_usmle_n200.csv --n 100 --scoring letter --max-new-tokens 40"
) else if /i "%JOB%"=="pubmedqa" (
  set "OUT=results\results_openrouter_pubmedqa.json"
  set "LOG=logs\openrouter_pubmedqa_full.log"
  set "EXITFILE=logs\openrouter_pubmedqa_full.exit.txt"
  set "ARGS=--benchmark custom --custom-csv benchmarks\pubmedqa_n200.csv --n 100 --scoring yesno --max-new-tokens 10"
) else if /i "%JOB%"=="medhallu" (
  set "OUT=results\results_openrouter_medhallu.json"
  set "LOG=logs\openrouter_medhallu_full.log"
  set "EXITFILE=logs\openrouter_medhallu_full.exit.txt"
  set "ARGS=--benchmark custom --custom-csv benchmarks\medhallu_n200.csv --n 100 --scoring cosine --max-new-tokens 80"
) else (
  echo Unknown job: %JOB%
  exit /b 4
)

echo Running job=%JOB% model=%MODEL%
echo Output=%OUT%
echo Log=%LOG%

"llm-env\Scripts\python.exe" -u cured.py %COMMON% %ARGS% --out "%OUT%" > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo exit=%RC% > "%EXITFILE%"
echo Finished job=%JOB% exit=%RC%

popd
exit /b %RC%
