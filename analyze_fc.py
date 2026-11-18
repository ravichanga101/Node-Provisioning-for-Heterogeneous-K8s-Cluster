"""Forecast-accuracy tables on the chronological test split (first 3 steps = 15-min horizon)."""
import json, numpy as np, pandas as pd, os
SCN = ['T1-CloudWatch-A', 'T2-CloudWatch-B', 'T3-Diurnal-Taxi', 'S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']
CL = ['arm_bound', 'x86_bound', 'portable']
rows = []; hyb = []
import glob
for scn in SCN:
    df = pd.read_csv(f'data/{scn}.csv'); n = len(df); a = int(n * .70); b = int(n * .85); H = 6; o0 = a - 1
    Z = {}
    for f in glob.glob(f'out/pred/{scn}__*.npy'):
        _, s, kind, seed = os.path.basename(f)[:-4].split('__')
        Z[f'{s}|lstm_{kind}|{seed}'] = np.load(f)
    if os.path.exists(f'out/{scn}_base.npz'): Z.update(dict(np.load(f'out/{scn}_base.npz')))
    if not Z: continue
    origins = np.arange(b - 1, n - H); off = origins - o0
    for c in CL:
        y = df[c].values; Y = np.stack([y[o + 1:o + 4] for o in origins])
        models = sorted({k.split('|')[1] for k in Z if k.startswith(c + '|')})
        for m in models:
            seeds = sorted({int(k.split('|')[2]) for k in Z if k.startswith(f'{c}|{m}|')})
            for s in seeds:
                P = Z[f'{c}|{m}|{s}'][off]
                if m == 'lstm_quantile':
                    P50 = P[:, :3, 0]; P90 = P[:, :3, 2]
                    e = Y - P90
                    rows.append(dict(scenario=scn, stream=c, model='lstm_quantile(q0.5)', seed=s, MAE=np.abs(P50 - Y).mean(),
                                     RMSE=np.sqrt(((P50 - Y) ** 2).mean()), under=(Y > P50).mean() * 100,
                                     pinball90=np.maximum(0.9 * e, -0.1 * e).mean(), cover90=(Y <= P90).mean() * 100))
                    continue
                P = P[:, :3]; e = Y - P
                rows.append(dict(scenario=scn, stream=c, model=m, seed=s, MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                                 under=(Y > P).mean() * 100, pinball90=np.maximum(0.9 * e, -0.1 * e).mean(), cover90=(Y <= P).mean() * 100))
            if m.startswith('lstm_hybrid'):
                Pm = Z[f'{c}|lstm_mae|0'][off][:, :3]; Ph = Z[f'{c}|{m}|0'][off][:, :3]
                hyb.append(dict(scenario=scn, stream=c, variant=m, max_abs_diff=np.abs(Pm - Ph).max(), mean_abs_diff=np.abs(Pm - Ph).mean(),
                                MAE_mae=np.abs(Pm - Y).mean(), MAE_hybrid=np.abs(Ph - Y).mean(),
                                k=0.7 + 0.3 * {'arm': 0.034, 'x86': 0.0425}[m.split('_')[-1]] * 5 / 60))
R = pd.DataFrame(rows); R.to_csv('out/fc_accuracy_raw.csv', index=False)
G = R.groupby(['scenario', 'stream', 'model']).agg(MAE=('MAE', 'mean'), MAE_sd=('MAE', 'std'), RMSE=('RMSE', 'mean'),
        under=('under', 'mean'), pinball90=('pinball90', 'mean'), cover90=('cover90', 'mean'), nseeds=('seed', 'count')).reset_index()
G.to_csv('out/fc_accuracy.csv', index=False)
if hyb: pd.DataFrame(hyb).to_csv('out/hybrid_equivalence.csv', index=False)
print(G.pivot_table(index=['scenario', 'stream'], columns='model', values='MAE').round(3).to_string())
if hyb: print(pd.DataFrame(hyb).round(5).to_string())
