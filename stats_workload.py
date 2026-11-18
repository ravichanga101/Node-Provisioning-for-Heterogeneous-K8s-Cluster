"""Workload-level paired analysis: seeds averaged within each workload (experimental unit = workload, n=9).
Reports per-workload differences, Wilcoxon on n=9, matched-pairs rank-biserial effect size, and
cluster-bootstrap 95% CIs (resampling workloads) for mean relative change."""
import numpy as np, pandas as pd, json
from scipy.stats import wilcoxon
m=pd.read_csv('out/sim_main.csv'); P='Proposed v2'
REF=['Reactive (Karpenter)','Threshold (70%, arch-agnostic)','v2 w/o quantile (point+eta)','v2 w/o portability','v2 w/o burst margin']
mets=['cost_per_day','unserved_vcpu_min','shortfall_pct']
g=m.groupby(['strategy','scenario'])[mets].mean()
rng=np.random.default_rng(0); out={}
def rbc(d):
    d=d[d!=0]
    if len(d)==0: return 0.0
    r=pd.Series(np.abs(d)).rank().values; return float((r[d>0].sum()-r[d<0].sum())/r.sum())
for ref in REF:
    for met in mets:
        a=g.loc[P][met]; b=g.loc[ref][met]; d=(a-b).values
        p=wilcoxon(a,b).pvalue if (d!=0).any() else 1.0
        # relative change of the workload-mean metric (ratio of sums over workloads), bootstrap over workloads
        if met=='cost_per_day':   # mean of per-workload relative changes (the definition used in the paper)
            def rel(ix): return float(np.mean(a.values[ix]/b.values[ix]-1))
        else:                     # absolute paired difference (relative change is undefined when the reference is ~0)
            def rel(ix): return float(np.mean(a.values[ix]-b.values[ix]))
        bs=[rel(rng.integers(0,9,9)) for _ in range(5000)]
        out[f'{ref}|{met}']=dict(lower_in=int((d<0).sum()),higher_in=int((d>0).sum()),p=float(p),rbc=rbc(d),
            rel=rel(np.arange(9)),ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
        print(f'{ref[:26]:26s} {met[:12]:12s} lower {int((d<0).sum())}/9 higher {int((d>0).sum())}/9 p={p:.3f} rbc={rbc(d):+.2f} eff={rel(np.arange(9)):+.3f} CI=[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]')
json.dump(out,open('out/stats_workload.json','w'),indent=1)
