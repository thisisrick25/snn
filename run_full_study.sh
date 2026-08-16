#!/usr/bin/env bash
# Linux/Kaggle equivalent of run_full_study.ps1: 2 settings x 3 mechanisms +
# 3 confound controls x 2 settings, all landing in results/full_study, then the
# confirmatory analysis. Resume-aware: a whole mechanism whose raw JSON count is
# already complete is skipped, and run_pilot's own --resume skips finished seeds
# within a partially-done mechanism. Run from the repo root: ./run_full_study.sh
# Optional first arg = thread cap to keep the machine usable, e.g. ./run_full_study.sh 4
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

config="configs/full_study.yaml"
raw="results/full_study/raw"
threads="${1:-0}"
thread_args=()
if [ "$threads" -gt 0 ]; then thread_args=(--threads "$threads"); fi

n_seeds=9

# Count raw JSONs already present for a (dataset, arch, mechanism, control) cell.
count_done() {
    local ds="$1" arch="$2" mech="$3" ctrl="$4"
    ls "$raw"/seed*_"${ds}"_"${arch}"_"${mech}"_"${ctrl}"_*.json 2>/dev/null | wc -l
}

run_pilot() {
    echo "==> python -m src.scripts.run_pilot $*"
    python -m src.scripts.run_pilot "$@"
}

# 1. Core matrix: 2 settings x 3 mechanisms. threshold sweeps 9 thetas, the
#    others 4 conditions each -> expected = n_seeds * n_conditions.
for setting in "mnist mlp" "cifar10 conv_snn"; do
    read -r ds arch <<< "$setting"
    for mech in kwta_window activity_reg threshold; do
        if [ "$mech" = "threshold" ]; then expected=$((n_seeds * 9)); else expected=$((n_seeds * 4)); fi
        have=$(count_done "$ds" "$arch" "$mech" none)
        if [ "$have" -ge "$expected" ]; then
            echo "[skip] $ds/$arch $mech complete ($have/$expected)"
        else
            run_pilot --config "$config" --dataset "$ds" --arch "$arch" --sparsity-mode "$mech" "${thread_args[@]}"
        fi
    done
done

# 2. Confound controls on a dense (threshold) reference: 3 controls x 2 settings.
for setting in "mnist mlp" "cifar10 conv_snn"; do
    read -r ds arch <<< "$setting"
    for ctrl in update_norm activation_dropout block_freeze; do
        expected=$n_seeds
        have=$(count_done "$ds" "$arch" threshold "$ctrl")
        if [ "$have" -ge "$expected" ]; then
            echo "[skip] $ds/$arch control=$ctrl complete ($have/$expected)"
        else
            run_pilot --config "$config" --dataset "$ds" --arch "$arch" --sparsity-mode threshold --control "$ctrl" "${thread_args[@]}"
        fi
    done
done

# 3. Confirmatory analysis over the accumulated summary.csv.
echo "==> python -m src.scripts.run_confirmatory --config $config"
python -m src.scripts.run_confirmatory --config "$config"

echo "Full study complete. Results in results/full_study."
