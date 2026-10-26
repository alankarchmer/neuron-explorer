#!/usr/bin/env bash
# Investing sweep: the invest_rel design over each layer's investing candidates (sweep/invest_candidates.json), with
# random controls per layer, then the forced-choice buy/sell lens over every neuron of those layers.
#     setsid nohup sweep/run_invest.sh "4 5 6 7 8 9" 64 </dev/null >/dev/null 2>&1 &
# With no layer list it runs every candidate layer from L3 up: at L0-L2 relative doses empty most rollouts, random
# directions included (research_log.md, depth survey), so they aren't measurements there. A third argument of 0 skips
# the choice lens (about 20 min per layer; at chance level in the L7/L24 pilot).
set -uo pipefail
LAYERS="${1:-}"; NCTRL="${2:-64}"; CHOICE="${3:-1}"
cd /workspace/neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p stats runs
CAND=sweep/invest_candidates.json
[ -n "$LAYERS" ] || LAYERS=$(python -c "import json; print(' '.join(l for l in json.load(open('$CAND'))['by_layer'] if int(l) >= 3))")
for l in $LAYERS; do
  # one layer per stats pass: a single pass over all 27 layers runs a 24 GB card out of memory
  [ -f "stats/q35-4b_L$l.npz" ] || python sweep/neuron_stats.py --layers "$l" --tokens 200000 --out stats/q35-4b \
    > "invest_stats_L$l.log" 2>&1 || echo "L$l stats FAILED $(date)" >> invest.log
  NEUR=$(python -c "import json; print(','.join(map(str, json.load(open('$CAND'))['by_layer']['$l'])))")
  echo "L$l start $(date)" >> invest.log
  python sweep/steer_sweep.py --stats "stats/q35-4b_L$l.npz" --design invest_rel --dose-unit rel --neurons "$NEUR" \
    --random-controls "$NCTRL" --max-num-seqs 320 --run "q35-4b-L$l-invest" > "sweep_L$l-invest.log" 2>&1 \
    && echo "L$l done $(date)" >> invest.log || echo "L$l FAILED $(date)" >> invest.log
done
if [ "$CHOICE" != 0 ]; then
  echo "choice lens start $(date)" >> invest.log
  python sweep/steer_lens.py --mode choice --layers "$(echo $LAYERS | tr ' ' ',')" --out stats/steer_choice \
    > choice.log 2>&1 && echo "choice lens done $(date)" >> invest.log || echo "choice lens FAILED $(date)" >> invest.log
fi
echo ALL_DONE >> invest.log
