#!/usr/bin/env bash
#
# Reproduce every number in the paper from the raw input.
#
#   bash run_all.sh          full pipeline, including the slow model fits
#   bash run_all.sh --fast   skip the neural stages (05, 05b, 05c, 06, 07, 09, 16, 19)
#
# The fast path takes about two minutes and reproduces the headline identification
# result, the economics, and every table. The full path adds the forecasting and
# compositional benchmarks and takes roughly 60-90 minutes on four CPU cores.
#
# Stage order matters: 01 -> 03 -> 04 must run before anything else, and 11 must
# run before 12 and 14.

set -euo pipefail
cd "$(dirname "$0")"

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

run () {
    printf '%-26s ' "$1"
    if python "src/$1.py" > "logs/$1.log" 2>&1; then
        echo "ok"
    else
        echo "FAILED  (see logs/$1.log)"
        tail -5 "logs/$1.log" | sed 's/^/    /'
        exit 1
    fi
}

mkdir -p logs paper/tabs paper/figtex paper/figs

echo "=== stage 1: data preparation ==="
run 01_clean          # physics-informed cleaning; writes data/hourly_clean.csv
run 03_weather        # NASA POWER download (needs network, ~2 min)
run 04_features       # design matrix

echo
echo "=== stage 2: identification and economics (fast) ==="
run 17_identification # partial identification + price/activity controls
run 08_economics      # loss-adjusted economics, counterfactual, sensitivity

if [ "$FAST" -eq 0 ]; then
  echo
  echo "=== stage 3: model fits (slow) ==="
  run 05_censored       # Tobit synthetic-censoring recovery
  run 05b_lambda        # penalty / anchor-size diagnosis
  run 05c_interval      # Tobit vs interval censoring, known truth
  run 19_sweep_magnitude# extended sweep, 0-4.9 pp
  run 16_alpha_sensitivity
  run 06_forecast       # demand forecasting benchmark
  run 07_composition    # compositional fuel-mix forecasting
  run 09_projection     # latent demand series
else
  echo
  echo "=== stage 3 skipped (--fast) ==="
  echo "    Tables 3, 5, 6 and 7 depend on results/ files shipped with the"
  echo "    repository; rerun without --fast to regenerate them."
fi

echo
echo "=== stage 4: figures and tables ==="
run 10_figures        # -> paper/figs/
run 11_tables         # -> paper/tables.tex
run 12_identity_table # appends the accounting-identity table
run 18_identification_table
run 14_split_floats   # one file per float, for placement
run 02_eda            # exploratory summary (not required by the paper)

echo
echo "=== done ==="
echo "Figures  : paper/figs/"
echo "Tables   : paper/tabs/"
echo "Logs     : logs/"
echo
echo "Headline numbers:"
python - <<'PY'
import json
try:
    R = json.load(open('results/identification.json'))
    g = lambda d, y: float(d.get(str(y), d.get(y)))
    print(f"  2023 suppression bounds : [{g(R['bounds']['lo_pct'],2023):.2f}, "
          f"{g(R['bounds']['hi_pct'],2023):.2f}] pp")
    print(f"  structural, controlled  : {g(R['gap_both_controls'],2023):+.2f} pp")
    print(f"  real tariff 2021->2025  : {R['real_tariff_change_2021_2025_pct']:+.1f} %")
except Exception as e:
    print('  (identification.json unavailable:', e, ')')
PY
