import json
from evaluation.experiments import run_multiple_splits, run_control_experiment, _progression
import pandas as pd

data = pd.read_csv('data/sample_data/payments.csv')
multi = run_multiple_splits(n_seeds=10)
progression = _progression(data)

cache = {
    'multi_split_avg_incremental_revenue': multi['average_incremental_revenue'],
    'multi_split_avg_incremental_rate_pp': multi['average_incremental_rate_pp'],
    'progression': {k: {kk: vv for kk, vv in v.items()} for k, v in progression.items()}
}
with open('evaluation/dashboard_cache.json', 'w') as f:
    json.dump(cache, f, indent=2)
print('saved evaluation/dashboard_cache.json')