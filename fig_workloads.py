import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
SCN = ['T1-CloudWatch-A', 'T2-CloudWatch-B', 'T3-Diurnal-Taxi', 'S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']
CL = {'arm_bound': ('ARM64-bound', '#2a6f97'), 'x86_bound': ('x86-64-bound', '#c8553d'), 'portable': ('Portable', '#5b8e3e')}
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0), sharex=True)
rows = []
for ax, scn in zip(axs.flat, SCN):
    df = pd.read_csv(f'data/{scn}.csv'); n = len(df); a = int(n * .7); b = int(n * .85)
    d = df.minute.values / 1440
    for c, (lab, col) in CL.items():
        ax.plot(d, df[c].values, lw=0.5, color=col, label=lab)
        y = df[c].values; s = pd.Series(y)
        rows.append(dict(scenario=scn, stream=lab, mean=y.mean(), sd=y.std(), cv=y.std() / y.mean(), min=y.min(), p50=np.median(y),
                         p99=np.percentile(y, 99), max=y.max(), acf1=s.autocorr(1), acf_day=s.autocorr(288),
                         adf_p=adfuller(y, maxlag=48, autolag=None)[1]))
    ax.axvspan(d[a], d[b], color='0.85', lw=0); ax.axvspan(d[b], d[-1], color='0.7', lw=0)
    ax.set_title(scn, fontsize=8); ax.set_ylim(bottom=0)
for ax in axs[-1]: ax.set_xlabel('Day')
for ax in axs[:, 0]: ax.set_ylabel('CPU request (vCPU)')
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=2) for _, c in CL.values()]; l = [v[0] for v in CL.values()]
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig('figs/workloads.pdf'); fig.savefig('figs/workloads.png', dpi=150)
R = pd.DataFrame(rows); R.to_csv('tables/workload_stats.csv', index=False)
print(R.round(2).to_string())
# zoom figure: one test day for 3 scenarios with 5-min detail

fig, axs = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True)
for ax, scn in zip(axs.flat, SCN):
    df = pd.read_csv(f'data/{scn}.csv'); n = len(df); b = int(n * .85)
    seg = df.iloc[b:b + 288]; hrs = (seg.minute.values - seg.minute.values[0]) / 60
    for c, (lab, col) in CL.items():
        ax.plot(hrs, seg[c].values, lw=0.8, color=col)
    ax.set_title(scn, fontsize=8); ax.set_ylim(bottom=0); ax.set_xticks([0, 6, 12, 18, 24])
for ax in axs[-1]: ax.set_xlabel('Hour of first test day')
for ax in axs[:, 0]: ax.set_ylabel('CPU request (vCPU)')
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig('figs/workloads_zoom.pdf'); fig.savefig('figs/workloads_zoom.png', dpi=150)
