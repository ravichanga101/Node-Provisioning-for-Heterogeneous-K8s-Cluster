"""Generate LaTeX table bodies (blue) from result CSVs."""
import pandas as pd, numpy as np, json, os
SCN = ['T1-CloudWatch-A', 'T2-CloudWatch-B', 'T3-Diurnal-Taxi', 'S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']
ABBR = {s: s.split('-')[0] for s in SCN}
out = {}
# workload statistics
W = pd.read_csv('tables/workload_stats.csv')
rows = []
for scn in SCN:
    for i, (_, r) in enumerate(W[W.scenario == scn].iterrows()):
        st = {'ARM64-bound': 'ARM', 'x86-64-bound': 'x86', 'Portable': 'Port.'}[r.stream]
        rows.append(f"{ABBR[scn] if i == 0 else ''} & {st} & {r['mean']:.2f} & {r.sd:.2f} & {r.cv:.2f} & {r['min']:.2f} & {r.p50:.2f} & {r['max']:.2f} & {r.acf1:.2f} & {r.acf_day:.2f} & {'$<$0.01' if r.adf_p < 0.01 else f'{r.adf_p:.2f}'} \\\\")
        if i == 2 and scn != SCN[-1]: rows.append('\\addlinespace[1pt]')
out['wstats'] = '\n'.join(rows)
# forecast accuracy (MAE, 15-min) per stream; LSTM mean over seeds
G = pd.read_csv('out/fc_accuracy.csv')
models = [('naive', 'Naive'), ('snaive', 'S-naive'), ('arima', 'ARIMA'), ('prophet', 'Prophet'), ('ridge', 'Ridge'), ('lstm_mae', 'LSTM (MAE)'), ('lstm_quantile(q0.5)', 'LSTM-Q (median)')]
rows = []; wins = {m: 0 for m, _ in models}
for scn in SCN:
    for i, c in enumerate(['arm_bound', 'x86_bound', 'portable']):
        g = G[(G.scenario == scn) & (G.stream == c)].set_index('model')
        vals = {m: g.loc[m, 'MAE'] for m, _ in models if m in g.index}
        best = min(vals.values()); wins[min(vals, key=vals.get)] += 1
        cells = []
        for m, _ in models:
            v = vals.get(m, np.nan); sd = g.loc[m, 'MAE_sd'] if m in g.index else np.nan
            s = f'{v:.3f}' if v < 10 else f'{v:.2f}'
            if m.startswith('lstm') and g.loc[m, 'nseeds'] > 1: s += f'$\\pm${sd:.3f}'
            if abs(v - best) < 1e-12: s = '\\textbf{' + s + '}'
            cells.append(s)
        st = {'arm_bound': 'ARM', 'x86_bound': 'x86', 'portable': 'Port.'}[c]
        rows.append(f"{ABBR[scn] if i == 0 else ''} & {st} & " + ' & '.join(cells) + ' \\\\')
        if i == 2 and scn != SCN[-1]: rows.append('\\addlinespace[1pt]')
out['fc'] = '\n'.join(rows); out['fc_wins'] = wins
# quantile calibration
Q = G[G.model == 'lstm_quantile(q0.5)']
out['q_cover_mean'] = float(Q.cover90.mean()); out['q_cover_min'] = float(Q.cover90.min()); out['q_cover_max'] = float(Q.cover90.max())
L = G[G.model == 'lstm_mae']; out['mae_under_mean'] = float(L.under.mean())
json.dump({k: v for k, v in out.items() if k not in ('wstats', 'fc')}, open('tables/fc_summary.json', 'w'), indent=1)
for k in ('wstats', 'fc'):
    open(f'tables/{k}.tex', 'w').write(out[k])
print(out['fc_wins'], out['q_cover_mean'], out['q_cover_min'], out['q_cover_max'], out['mae_under_mean'])
