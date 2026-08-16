# Runs the full study sequentially: 2 settings x 3 mechanisms + 3 confound
# controls x 2 settings, all landing in results/full_study, then the
# confirmatory analysis. Run from the repo root: .\run_full_study.ps1
# Optional: pass a thread cap to keep the desktop usable, e.g. .\run_full_study.ps1 -Threads 4

param(
    [int]$Threads = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = $PSScriptRoot

$config = "configs/full_study.yaml"
$raw = "results/full_study/raw"
$nSeeds = 9
$threadArgs = @()
if ($Threads -gt 0) { $threadArgs = @("--threads", $Threads) }

function Get-DoneCount {
    param([string]$Dataset, [string]$Arch, [string]$Mech, [string]$Ctrl)
    (Get-ChildItem -Path $raw -Filter "seed*_${Dataset}_${Arch}_${Mech}_${Ctrl}_*.json" -ErrorAction SilentlyContinue).Count
}

# setting = dataset + arch pair; each core run sweeps one mechanism's grid.
$settings = @(
    @{ dataset = "mnist";   arch = "mlp" },
    @{ dataset = "cifar10"; arch = "conv_snn" }
)
$mechanisms = @("kwta_window", "activity_reg", "threshold")
$controls   = @("update_norm", "activation_dropout", "block_freeze")

function Invoke-Run {
    param([string[]]$RunArgs)
    Write-Host "==> python -m src.scripts.run_pilot $($RunArgs -join ' ')" -ForegroundColor Cyan
    python -m src.scripts.run_pilot @RunArgs
    if ($LASTEXITCODE -ne 0) { throw "run_pilot failed (exit $LASTEXITCODE): $($RunArgs -join ' ')" }
}

# 1. Core matrix: 2 settings x 3 mechanisms. threshold sweeps 9 thetas, the
#    others 4 conditions each; a fully-populated mechanism is skipped.
foreach ($s in $settings) {
    foreach ($mech in $mechanisms) {
        $expected = if ($mech -eq "threshold") { $nSeeds * 9 } else { $nSeeds * 4 }
        $have = Get-DoneCount $s.dataset $s.arch $mech "none"
        if ($have -ge $expected) {
            Write-Host "[skip] $($s.dataset)/$($s.arch) $mech complete ($have/$expected)" -ForegroundColor Yellow
        } else {
            Invoke-Run (@("--config", $config, "--dataset", $s.dataset, "--arch", $s.arch, "--sparsity-mode", $mech) + $threadArgs)
        }
    }
}

# 2. Confound controls on a dense (threshold) reference: 3 controls x 2 settings.
foreach ($s in $settings) {
    foreach ($ctrl in $controls) {
        $have = Get-DoneCount $s.dataset $s.arch "threshold" $ctrl
        if ($have -ge $nSeeds) {
            Write-Host "[skip] $($s.dataset)/$($s.arch) control=$ctrl complete ($have/$nSeeds)" -ForegroundColor Yellow
        } else {
            Invoke-Run (@("--config", $config, "--dataset", $s.dataset, "--arch", $s.arch, "--sparsity-mode", "threshold", "--control", $ctrl) + $threadArgs)
        }
    }
}

# 3. Confirmatory analysis over the accumulated summary.csv.
Write-Host "==> python -m src.scripts.run_confirmatory --config $config" -ForegroundColor Green
python -m src.scripts.run_confirmatory --config $config
if ($LASTEXITCODE -ne 0) { throw "run_confirmatory failed (exit $LASTEXITCODE)" }

Write-Host "Full study complete. Results in results/full_study." -ForegroundColor Green
