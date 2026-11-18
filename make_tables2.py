import pandas as pd, numpy as np, json, sys
sys.path.insert(0, '.')
from analyze_sim import ORDER, SHORT, SCN, ci
m = pd.read_csv('out/sim_main.csv')
base = m[m.strategy == 'Reactive (Karpenter)'].set_index(['scenario', 'seed'])
thr = m[m.strategy == 'Threshold (70%, arch-agnostic)'].set_index(['scenario', 'seed'])
def rel(df, ref, col):
    return df.apply(lambda r: r[col] / ref.loc[(r.scenario, r.seed % 10 if (r.scenario, r.seed) not in ref.index else r.seed)][col], axis=1)
m['cost_rel'] = m.apply(lambda r: r.cost_per_day / base.loc[(r.scenario, r.seed)].cost_per_day, axis=1)
m['cost_rel_thr'] = m.apply(lambda r: r.cost_per_day / thr.loc[(r.scenario, r.seed)].cost_per_day, axis=1)
N = {}
S = m.groupby('strategy').agg(cost=('cost_per_day', 'mean'), cost_rel=('cost_rel', 'mean'), short=('shortfall_pct', 'mean'),
                              uns=('unserved_vcpu_min', 'mean'), slo=('sla_viol_pct', 'mean'), alloc=('util_pct', 'mean'),
                              eps=('pend_episodes_per_day', 'mean'), epm=('pend_ep_mean_s', 'mean'), launch=('node_launches_per_day', 'mean'),
                              arm=('cost_arm_share', 'mean')).reindex(ORDER)
# unserved reduction vs reactive: computed on scenario means
sm = m.groupby(['scenario', 'strategy']).unserved_vcpu_min.mean().unstack()
red = (1 - sm.div(sm['Reactive (Karpenter)'], axis=0)) * 100
S['uns_red'] = red.mean().reindex(ORDER)
rows = []
for k, r in S.iterrows():
    nm = SHORT[k]; b = '\\textbf{' if k == 'Proposed v2' else ''; e = '}' if b else ''
    rows.append(f"{b}{nm}{e} & {r.cost:.2f} & {100*(r.cost_rel-1):+.1f} & {r.short:.2f} & {r.uns:.2f} & {r.uns_red:.0f} & {r.slo:.2f} & {r.alloc:.1f} & {r.eps:.1f} & {r.launch:.1f} \\\\")
open('tables/main.tex', 'w').write('\n'.join(rows))
# per-scenario detail: cost and unserved with CI for key strategies
A = m.groupby(['scenario', 'strategy']).agg(c=('cost_per_day', 'mean'), cci=('cost_per_day', ci), u=('unserved_vcpu_min', 'mean'), uci=('unserved_vcpu_min', ci),
                                            s=('shortfall_pct', 'mean')).reset_index()
keys = ['Reactive (Karpenter)', 'Threshold (70%, arch-agnostic)', 'v2 w/o quantile (point+eta)', 'Proposed v2', 'Oracle (perfect forecast)']
rows = []
for scn in SCN:
    cells = []
    for k in keys:
        r = A[(A.scenario == scn) & (A.strategy == k)].iloc[0]
        cells.append(f"{r.c:.1f} / {r.u:.1f}")
    rows.append(scn.split('-')[0] + ' & ' + ' & '.join(cells) + ' \\\\')
open('tables/per_scenario.tex', 'w').write('\n'.join(rows))
# full per-scenario appendix table with CI for all strategies
rows = []
for scn in SCN:
    for i, k in enumerate(ORDER):
        r = A[(A.scenario == scn) & (A.strategy == k)].iloc[0]
        rows.append(f"{scn.split('-')[0] if i == 0 else ''} & {SHORT[k]} & {r.c:.2f} $\\pm$ {r.cci:.2f} & {r.s:.2f} & {r.u:.2f} $\\pm$ {r.uci:.2f} \\\\")
    if scn != SCN[-1]: rows.append('\\midrule')
open('tables/appendix_full.tex', 'w').write('\n'.join(rows))
# headline numbers
v2 = m[m.strategy == 'Proposed v2']
N['v2_cost_vs_reactive'] = 100 * (v2.cost_rel.mean() - 1)
N['v2_cost_vs_threshold'] = 100 * (v2.cost_rel_thr.mean() - 1)
N['v2_uns_red'] = S.loc['Proposed v2', 'uns_red']; N['thr_uns_red'] = S.loc['Threshold (70%, arch-agnostic)', 'uns_red']
N['short'] = S.short.to_dict(); N['uns'] = S.uns.to_dict(); N['cost_rel'] = (100 * (S.cost_rel - 1)).to_dict()
N['port_saving'] = 100 * (1 - m[m.strategy == 'Proposed v2'].cost_per_day.mean() / m[m.strategy == 'v2 w/o portability'].cost_per_day.mean())
# tau table
t = pd.read_csv('out/sim_tau.csv')
t['cost_rel'] = t.apply(lambda r: r.cost_per_day / base.loc[(r.scenario, r.seed)].cost_per_day, axis=1)
T = t.groupby('strategy').agg(cost=('cost_per_day', 'mean'), cost_rel=('cost_rel', 'mean'), short=('shortfall_pct', 'mean'), uns=('unserved_vcpu_min', 'mean'), slo=('sla_viol_pct', 'mean'))
rows = []
for k, r in T.iterrows():
    tau = float(k.split('=')[1]); rho = tau / (1 - tau)
    rows.append(f"{tau:.2f} & {rho:.0f} & {r.cost:.2f} & {100*(r.cost_rel-1):+.1f} & {r.short:.3f} & {r.uns:.2f} & {r.slo:.3f} \\\\")
for k in ['Threshold (70%, arch-agnostic)', 'v2 w/o quantile (point+eta)']:
    r = S.loc[k]; rows.append(f"\\multicolumn{{2}}{{l}}{{{SHORT[k]}}} & {r.cost:.2f} & {100*(r.cost_rel-1):+.1f} & {r.short:.3f} & {r.uns:.2f} & {r.slo:.3f} \\\\")
open('tables/tau.tex', 'w').write('\n'.join(rows)); N['tau'] = T.to_dict()
# horizon
h = pd.read_csv('out/sim_horizon.csv'); h['cost_rel'] = h.apply(lambda r: r.cost_per_day / base.loc[(r.scenario, r.seed)].cost_per_day, axis=1)
Hh = h.groupby('strategy').agg(cost=('cost_per_day', 'mean'), cost_rel=('cost_rel', 'mean'), short=('shortfall_pct', 'mean'), uns=('unserved_vcpu_min', 'mean'))
Hh = Hh.loc[['H=5min', 'H=10min', 'H=15min', 'H=30min']]
open('tables/horizon.tex', 'w').write('\n'.join(f"{k.split('=')[1].replace('min',' min')} & {r.cost:.2f} & {100*(r.cost_rel-1):+.1f} & {r.short:.3f} & {r.uns:.2f} \\\\" for k, r in Hh.iterrows()))
N['horizon'] = Hh.to_dict()
# kappa
kp = pd.read_csv('out/sim_kappa.csv')
K = kp.groupby(['cfg_kappa', 'strategy']).agg(cost=('cost_per_day', 'mean'), arm=('cost_arm_share', 'mean'), uns=('unserved_vcpu_min', 'mean')).reset_index()
rows = []
for kk in (0.8, 1.0, 1.25, 1.5):
    g = K[K.cfg_kappa == kk].set_index('strategy')
    v, nop, re = g.loc['Proposed v2'], g.loc['v2 w/o portability'], g.loc['Reactive (Karpenter)']
    rows.append(f"{kk:.2f} & {0.034*kk:.4f} & {v.cost:.2f} & {nop.cost:.2f} & {100*(1-v.cost/nop.cost):.1f} & {100*v.arm:.0f} & {100*nop.arm:.0f} & {re.cost:.2f} \\\\")
open('tables/kappa.tex', 'w').write('\n'.join(rows)); N['kappa'] = K.to_dict()
# scale
sc = pd.read_csv('out/sim_scale.csv'); bs = sc[sc.strategy == 'Reactive (Karpenter)'].set_index(['scenario', 'seed'])
sc['cost_rel'] = sc.apply(lambda r: r.cost_per_day / bs.loc[(r.scenario, r.seed)].cost_per_day, axis=1)
ss = sc.groupby(['scenario', 'strategy']).unserved_vcpu_min.mean().unstack(); sred = (1 - ss.div(ss['Reactive (Karpenter)'], axis=0)) * 100
SC = sc.groupby('strategy').agg(cost=('cost_per_day', 'mean'), cost_rel=('cost_rel', 'mean'), short=('shortfall_pct', 'mean'), uns=('unserved_vcpu_min', 'mean'),
                               slo=('sla_viol_pct', 'mean'), alloc=('util_pct', 'mean'), launch=('node_launches_per_day', 'mean'))
SC['uns_red'] = sred.mean()
SC = SC.loc[['Reactive (Karpenter)', 'Threshold (70%, arch-agnostic)', 'Proposed v2']]
open('tables/scale.tex', 'w').write('\n'.join(f"{SHORT[k]} & {r.cost:.1f} & {100*(r.cost_rel-1):+.1f} & {r.short:.2f} & {r.uns:.1f} & {r.uns_red:.0f} & {r.alloc:.1f} & {r.launch:.0f} \\\\" for k, r in SC.iterrows()))
N['scale'] = SC.to_dict()
# boot
b = pd.read_csv('out/sim_boot.csv'); bh = pd.read_csv('out/sim_bootH.csv')
N['boot'] = b.groupby(['strategy', 'cfg_boot_median'])[['shortfall_pct', 'unserved_vcpu_min', 'cost_per_day']].mean().reset_index().to_dict('records')
N['bootH'] = bh.groupby('cfg_boot_median')[['shortfall_pct', 'unserved_vcpu_min', 'cost_per_day']].mean().reset_index().to_dict('records')
N['overhead'] = json.load(open('out/overhead.json')); N['hybrid'] = json.load(open('out/hybrid_check.json'))
json.dump(N, open('tables/numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: N[k] for k in ['v2_cost_vs_reactive', 'v2_cost_vs_threshold', 'v2_uns_red', 'thr_uns_red', 'port_saving']}, indent=1, default=float))
print(open('tables/main.tex').read()); print(open('tables/tau.tex').read()); print(open('tables/horizon.tex').read()); print(open('tables/kappa.tex').read()); print(open('tables/scale.tex').read())
