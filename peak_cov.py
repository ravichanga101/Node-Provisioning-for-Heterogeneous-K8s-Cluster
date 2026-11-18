"""Empirical coverage of the horizon-peak rule max_i q_tau(t+i) for the 15-min peak, and per-step calibration per quantile."""
import json, glob, os, numpy as np, pandas as pd
SCN = ['T1-CloudWatch-A','T2-CloudWatch-B','T3-Diurnal-Taxi','S1-Constant','S2-Step','S3-Noisy','S4-Batch','S5-Spikes','S6-Sinusoid']
CL = ['arm_bound','x86_bound','portable']; Q=[0.5,0.8,0.9,0.95,0.99]
rows=[]
for scn in SCN:
    df=pd.read_csv(f'data/{scn}.csv'); n=len(df); a=int(n*.7); b=int(n*.85); H=6; o0=a-1
    origins=np.arange(b-1,n-H); off=origins-o0
    for c in CL:
        y=df[c].values; Y=np.stack([y[o+1:o+4] for o in origins]); Ypk=Y.max(1)
        for f in sorted(glob.glob(f'out/pred/{scn}__{c}__quantile__*.npy')):
            P=np.load(f)[off][:, :3, :]
            for k,q in enumerate(Q):
                rows.append(dict(scenario=scn,stream=c,seed=int(f[-5]),q=q,
                    step_cov=(Y<=P[:,:,k]).mean()*100, peak_cov=(Ypk<=P[:,:,k].max(1)).mean()*100))
d=pd.DataFrame(rows); d.to_csv('out/peak_cov.csv',index=False)
g=d.groupby(['q']).agg(step=('step_cov','mean'),peak=('peak_cov','mean')).round(1); print(g)
print(d[d.q==0.9].groupby('scenario')[['step_cov','peak_cov']].mean().round(1))
print(d[d.q==0.9].groupby(['scenario','stream'])['peak_cov'].mean().round(1).sort_values().head(6))
print('min/max peak cov q0.9 per scenario:',d[d.q==0.9].groupby('scenario').peak_cov.mean().agg(['min','max']).round(1).tolist())
json.dump(dict(by_q=g.reset_index().to_dict('records'),
  by_scn=d[d.q==0.9].groupby('scenario')[['step_cov','peak_cov']].mean().round(1).reset_index().to_dict('records')),open('out/peak_cov.json','w'),indent=1)
