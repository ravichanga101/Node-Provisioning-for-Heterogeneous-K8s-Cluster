import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.'); import experiments as E, sim
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
# 1) time series: T3 and S6, first 24 h of test, ARM pool, reactive vs proposed
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True)
for col, scn in enumerate(['T3-Diurnal-Taxi', 'S6-Sinusoid']):
    dem, fc = E.load(scn)
    for row, nm in enumerate(['Reactive (Karpenter)', 'Proposed v2']):
        r, rec = sim.run(dem, fc, E.STRAT[nm], seed=0, record=True)
        t = np.array(rec['t']) / 3600; m = t <= 24
        ax = axs[row, col]
        need = np.array(rec['d_arm'])
        ax.plot(t[m], np.array(rec['cap_arm'])[m], color='#2a6f97', lw=0.9, label='Ready allocatable capacity, ARM64 pool')
        ax.plot(t[m], need[m], color='#c8553d', lw=0.5, label='Demand on ARM64 pool')
        pend = np.array(rec['pend'])[m]
        ax.fill_between(t[m], 0, pend, color='#e0b252', lw=0.6, label='Pending, all pools')
        ax.set_title(f'{scn} — ' + ('Reactive' if nm.startswith('Reactive') else 'Proposed'), fontsize=8)
        ax.set_ylabel('vCPU'); ax.set_ylim(bottom=0)
for ax in axs[-1]: ax.set_xlabel('Hour of test segment')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig('figs/timeseries.pdf'); fig.savefig('figs/timeseries.png', dpi=150)
# 2) boot-time sensitivity
b = pd.read_csv('out/sim_boot.csv'); bh = pd.read_csv('out/sim_bootH.csv')
G = b.groupby(['strategy', 'cfg_boot_median'])[['shortfall_pct', 'unserved_vcpu_min', 'cost_per_day']].mean().reset_index()
GH = bh.groupby('cfg_boot_median')[['shortfall_pct', 'unserved_vcpu_min', 'cost_per_day']].mean().reset_index()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
sty = {'Reactive (Karpenter)': ('#c8553d', 'Reactive'), 'Threshold (70%, arch-agnostic)': ('#8d6a9f', 'Threshold'), 'Proposed v2': ('#2a6f97', 'Proposed (H = 15 min)')}
for k, (c, lab) in sty.items():
    g = G[G.strategy == k]
    axs[0].plot(g.cfg_boot_median, g.shortfall_pct, '-o', color=c, ms=3, label=lab)
    axs[1].plot(g.cfg_boot_median, g.unserved_vcpu_min, '-o', color=c, ms=3, label=lab)
axs[0].plot(GH.cfg_boot_median, GH.shortfall_pct, '--^', color='#2a6f97', ms=3, label='Proposed (H = 30 min)')
axs[1].plot(GH.cfg_boot_median, GH.unserved_vcpu_min, '--^', color='#2a6f97', ms=3, label='Proposed (H = 30 min)')
axs[0].set_ylabel('Shortfall time (%)'); axs[1].set_ylabel('Unserved (vCPU·min/day)')
for ax in axs: ax.set_xlabel('Median node boot time (s)'); ax.set_xticks([60, 180, 420, 600])
axs[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig('figs/boot.pdf'); fig.savefig('figs/boot.png', dpi=150)
print(G.round(2).to_string()); print(GH.round(2).to_string())
