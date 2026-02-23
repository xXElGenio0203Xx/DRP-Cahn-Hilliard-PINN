#!/bin/bash
# =============================================================================
# collect_results.sh — Aggregate summaries from all 17 experiments
# =============================================================================
# Run after all jobs complete to create a combined results table.
#
# Usage:
#   bash collect_results.sh
#
# Output:
#   results_summary.csv — one row per experiment
# =============================================================================

echo "experiment,optimizer,adam_epochs,bfgs_iters,adam_time_s,bfgs_time_s,total_time_s,final_adam_loss,final_bfgs_loss,n_params" > results_summary.csv

for summary_file in results_*/summary.json; do
    if [ ! -f "$summary_file" ]; then
        continue
    fi
    python3 -c "
import json, sys
with open('$summary_file') as f:
    s = json.load(f)
print(','.join([
    str(s.get('experiment', '')),
    str(s.get('optimizer', '')),
    str(s.get('adam_epochs', '')),
    str(s.get('bfgs_iters', '')),
    str(s.get('adam_time_s', '')),
    str(s.get('bfgs_time_s', '')),
    str(s.get('total_time_s', '')),
    str(s.get('final_adam_loss', '')),
    str(s.get('final_bfgs_loss', '')),
    str(s.get('n_params', '')),
]))
"
done >> results_summary.csv

echo "Results collected into results_summary.csv"
echo ""
column -t -s',' results_summary.csv
