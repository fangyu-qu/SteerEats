#!/usr/bin/env python3
"""Fill local report from a completed run with the actual processed input hash."""
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/logistic_ranker'
source = root / 'data/processed/sample_interactions.parquet'
metadata = json.loads((out / 'run_metadata.json').read_text())
if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['input_sha256']:
    raise SystemExit('Result/input hash mismatch: refusing to edit report')
data = pd.read_parquet(source)
# Existing data claims describe this development sample; do not silently mix cohorts.
if len(data) != 422788 or data.user_id.nunique() != 20000 or data.gmap_id.nunique() != 21155:
    raise SystemExit('Sample differs from the report cohort. Update data counts and provenance '
                     'before inserting these results; report left unchanged.')
metrics = pd.read_csv(out / 'metrics.csv')
test = metrics[metrics.period == 'test'].set_index('model')
if not {'Popularity', 'Recent popularity', 'Item-CF', 'Logistic regression'} <= set(test.index):
    raise SystemExit('Incomplete test model comparison')
if test.users.nunique() != 1:
    raise SystemExit('Model comparisons use different user cohorts')
parts = []
for label, name in [('Item-CF', 'Item-CF'), ('all-history popularity', 'Popularity'),
                    ('12-month popularity', 'Recent popularity'),
                    ('logistic regression', 'Logistic regression')]:
    r = test.loc[name]
    parts.append(f"{label} {r['Recall@10']:.4f} / {r['NDCG@10']:.4f}")
text = (r'\textbf{Baselines and learned ranker.} On a common historical catalog '
        f"({int(test.users.iloc[0]):,} test users), Recall@10 / NDCG@10 are: " +
        '; '.join(parts) + '. Features use historical data only; regularization was '
        f"selected on validation ($C={metadata['selected_C']:g}$). "
        f"We exclude {metadata['test_audit']['cold_positive_items']:,} cold-start positive interactions.")
report = root / 'report/main.tex'
old = report.read_text()
new, n = re.subn(r'\\textbf\{Baselines(?: and learned ranker)?\.\}.*?(?=\n\n)',
                 lambda _: text, old, count=1, flags=re.S)
if n != 1:
    raise SystemExit('Could not locate baseline paragraph')
backup = report.with_name('main.before_results.tex')
if not backup.exists():
    backup.write_text(old)
report.write_text(new)
print('Updated local report from genuine run files. Recompile and verify one page.')
