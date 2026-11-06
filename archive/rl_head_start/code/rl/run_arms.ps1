# Arms of the GRPO experiment in PLAN_2026-11-03_rl_head_start.md (see its 2026-11-03 amendment), run one after another on the GPU.
#
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','rl\run_arms.ps1' -WindowStyle Hidden
#   powershell -File rl\run_arms.ps1 -Runs "A1,B1" -DryRun     # print the commands only
#
# -Runs: arm letter + seed, in order. Arm X seed S goes to runs/grpo/armX_sS_<Steps>/, with stdout/stderr in
# runs/grpo/armX_sS_<Steps>.out/.err. Progress: runs/grpo/arms.log. A finished arm is skipped; a partial one is
# left alone (move it aside to rerun it).
param(
    [string]$Runs = "A0,B0,C0,D0,A1,B1",
    [int]$Steps = 100,
    [string]$EvalAt = "15,30,50,100",
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$progress = Join-Path $root "runs\grpo\arms.log"
$extra = @{
    A = @()
    B = @("--init", "runs/agent_dry/model_edit")
    C = @("--init", "runs/agent_dry/model_random")
    D = @("--temperature", "0.8")
}
function Note($msg) {
    $line = "$(Get-Date -Format s) $msg"
    if ($DryRun) { Write-Output $line } else { Add-Content $progress $line }
}
Note "driver pid $PID runs=$Runs steps=$Steps eval_at=$EvalAt"
foreach ($run in $Runs.Split(",")) {
    $arm = $run.Substring(0, 1); $seed = $run.Substring(1)
    $name = "arm${arm}_s${seed}_$Steps"
    $dir = Join-Path $root "runs\grpo\$name"
    $log = Join-Path $dir "log.jsonl"
    if ((Test-Path $log) -and ((Get-Content $log | Measure-Object -Line).Lines -ge $Steps)) {
        Note "$name already complete, skipping"; continue
    }
    if (Test-Path $dir) {
        Note "$name has a partial folder; refusing to append to it. Move it aside and rerun."; continue
    }
    $argv = @("rl/grpo.py", "--name", $name, "--steps", "$Steps", "--eval-at", $EvalAt, "--seed", $seed, "--trunc-zero") + $extra[$arm]
    Note "start $name : $($argv -join ' ')"
    if ($DryRun) { continue }
    $p = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "runs\grpo\$name.out" -RedirectStandardError "runs\grpo\$name.err"
    Note "end $name exit=$($p.ExitCode)"
}
Note "driver done"
