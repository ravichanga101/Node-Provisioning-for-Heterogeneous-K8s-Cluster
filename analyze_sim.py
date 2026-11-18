import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib, sys, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
SCN = ['T1-CloudWatch-A', 'T2-CloudWatch-B', 'T3-Diurnal-Taxi', 'S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']
ORDER = ['Reactive (Karpenter)', 'Threshold (70%, arch-agnostic)', 'Proposed v2',
         'v2 w/o quantile (point+eta)', 'v2 w/o portability', 'v2 w/o burst margin', 'Oracle (perfect forecast)']
SHORT = {'Reactive (Karpenter)': 'Reactive', 'Threshold (70%, arch-agnostic)': 'Threshold',
         'Proposed v2': 'Proposed', 'v2 w/o quantile (point+eta)': 'w/o quantile (point + head-room)', 'v2 w/o portability': 'w/o portability',
         'v2 w/o burst margin': 'w/o peak factor', 'Oracle (perfect forecast)': 'Oracle'}
COL = {'Reactive': '#c8553d', 'Threshold': '#8d6a9f', 'Proposed': '#2a6f97',
       'w/o quantile (point + head-room)': '#69a2b0', 'w/o portability': '#a9c5a0', 'w/o peak factor': '#e0b252', 'Oracle': '#222222'}
MET = ['cost_per_day', 'shortfall_pct', 'unserved_vcpu_min', 'sla_viol_pct', 'util_pct', 'pend_episodes_per_day', 'pend_ep_mean_s', 'node_launches_per_day', 'cost_arm_share']
def ci(x):
    x = np.asarray(x); n = len(x)
    return 0.0 if n < 2 else stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
def agg(df, keys):
    g = df.groupby(keys)
    out = g[MET].mean()
    for m in ('cost_per_day', 'shortfall_pct', 'unserved_vcpu_min'):
        out[m + '_ci'] = g[m].apply(ci)
    return out.reset_index()

def main_tables():
    df = pd.read_csv('out/sim_main.csv'); df = df[df.strategy.isin(ORDER)]; df['S'] = df.strategy.map(SHORT)
    A = agg(df, ['scenario', 'strategy']); A['S'] = A.strategy.map(SHORT); A.to_csv('tables/main_by_scenario.csv', index=False)
    # relative to reactive per scenario & seed (paired)
    base = df[df.strategy == 'Reactive (Karpenter)'].set_index(['scenario', 'seed'])
    rel = []
    for _, r in df.iterrows():
        b = base.loc[(r.scenario, r.seed)]
        rel.append(dict(scenario=r.scenario, strategy=r.strategy, seed=r.seed,
                        cost_rel=100 * (r.cost_per_day / b.cost_per_day - 1),
                        unserved_red=100 * (1 - r.unserved_vcpu_min / b.unserved_vcpu_min) if b.unserved_vcpu_min > 0 else np.nan))
    R = pd.DataFrame(rel); R.to_csv('tables/main_relative_raw.csv', index=False)
    S = df.groupby('strategy')[MET].mean().reindex(ORDER)
    S['cost_vs_reactive_pct'] = R.groupby('strategy').cost_rel.mean().reindex(ORDER)
    S['unserved_red_pct'] = R.groupby(['strategy', 'scenario']).unserved_red.mean().groupby('strategy').mean().reindex(ORDER)
    S.to_csv('tables/main_summary.csv'); print(S.round(2).to_string())
    # cost vs unserved scatter per scenario
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.4))
    for ax, scn in zip(axs.flat, SCN):
        sub = A[A.scenario == scn]
        for _, r in sub.iterrows():
            ax.errorbar(r.cost_per_day, r.unserved_vcpu_min, xerr=r.cost_per_day_ci, yerr=r.unserved_vcpu_min_ci, fmt='o', ms=4,
                        color=COL[r.S], label=r.S, capsize=1.5, lw=0.8)
        ax.set_title(scn, fontsize=8); ax.set_xlabel('Cost (USD/day)'); ax.set_ylabel('Unserved (vCPU·min/day)')
        ax.set_yscale('symlog', linthresh=1); ax.set_ylim(bottom=0)
    h, l = axs[0, 0].get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l): seen.setdefault(ll, hh)
    keys = [SHORT[k] for k in ORDER if SHORT[k] in seen]
    fig.legend([seen[k] for k in keys], keys, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig('figs/cost_vs_unserved.pdf'); fig.savefig('figs/cost_vs_unserved.png', dpi=150)
    return A, S

def tau_fig():
    df = pd.read_csv('out/sim_tau.csv'); m = pd.read_csv('out/sim_main.csv')
    rb = m[m.strategy == 'Reactive (Karpenter)'].groupby('scenario').cost_per_day.mean()
    tb = m[m.strategy == 'Threshold (70%, arch-agnostic)'].groupby('scenario')[['cost_per_day', 'shortfall_pct', 'unserved_vcpu_min']].mean()
    df['cost_rel'] = df.apply(lambda r: r.cost_per_day / rb[r.scenario], axis=1)
    T = df.groupby(['scenario', 'strategy'])[['cost_per_day', 'cost_rel', 'shortfall_pct', 'unserved_vcpu_min', 'sla_viol_pct']].mean().reset_index()
    T.to_csv('tables/tau_by_scenario.csv', index=False)
    G = T.groupby('strategy')[['cost_rel', 'shortfall_pct', 'unserved_vcpu_min', 'sla_viol_pct']].mean(); G.to_csv('tables/tau_summary.csv'); print(G.round(3))
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0))
    for ax, scn in zip(axs.flat, SCN):
        s = T[T.scenario == scn].sort_values('strategy')
        ax.plot(s.cost_per_day, s.shortfall_pct, '-o', color=COL['Proposed'], ms=3, lw=1, label='Proposed, $\\tau$ = 0.5 … 0.99')
        for _, r in s.iterrows(): ax.annotate(r.strategy.split('=')[1], (r.cost_per_day, r.shortfall_pct), fontsize=5.5, xytext=(2, 2), textcoords='offset points')
        mm = m[m.scenario == scn].groupby('strategy')[['cost_per_day', 'shortfall_pct']].mean()
        for k in ('Reactive (Karpenter)', 'Threshold (70%, arch-agnostic)', 'v2 w/o quantile (point+eta)'):
            ax.plot(mm.loc[k, 'cost_per_day'], mm.loc[k, 'shortfall_pct'], 's', ms=4, color=COL[SHORT[k]], label=SHORT[k])
        ax.set_title(scn, fontsize=8); ax.set_xlabel('Cost (USD/day)'); ax.set_ylabel('Shortfall time (%)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('figs/tau_tradeoff.pdf'); fig.savefig('figs/tau_tradeoff.png', dpi=150)

def simple(exp, keys):
    df = pd.read_csv(f'out/sim_{exp}.csv')
    T = df.groupby(keys)[['cost_per_day', 'shortfall_pct', 'unserved_vcpu_min', 'sla_viol_pct', 'util_pct', 'pend_ep_mean_s', 'node_launches_per_day', 'cost_arm_share', 'wall_s']].mean().reset_index()
    T.to_csv(f'tables/{exp}.csv', index=False); return T

if __name__ == '__main__':
    for w in sys.argv[1:]:
        if w == 'main': main_tables()
        if w == 'tau': tau_fig()
