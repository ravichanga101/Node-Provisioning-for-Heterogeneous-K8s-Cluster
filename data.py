"""Build workload scenarios (vCPU demand per class, 5-min resolution, 14 days).
Real-trace scenarios use public NAB traces (AWS CloudWatch EC2/RDS CPU, NYC taxi demand).
Synthetic scenarios follow the profiles described in the manuscript (Section 4)."""
import numpy as np, pandas as pd, sys, os
NAB = sys.argv[1]; OUT = sys.argv[2]
N = 4032  # 14 days x 288 five-minute steps
TARGET_P99 = {'arm_bound': 12.0, 'x86_bound': 12.0, 'portable': 16.0}
CLASSES = list(TARGET_P99)

def scale(v, cls):
    v = np.asarray(v, float)[:N]
    return v / np.percentile(v, 99) * TARGET_P99[cls]

def nab(name):
    return pd.read_csv(os.path.join(NAB, name + '.csv')).value.values

def taxi_window(start):
    d = pd.read_csv(os.path.join(NAB, 'nyc_taxi.csv'), parse_dates=['timestamp']).set_index('timestamp')
    w = d.loc[start:].iloc[:14 * 48 + 1].value
    w = w.resample('5min').interpolate('linear')
    return w.values[:N]

scen = {}
scen['T1-CloudWatch-A'] = {'arm_bound': nab('rds_cpu_utilization_cc0c53'),
                           'x86_bound': nab('ec2_cpu_utilization_ac20cd'),
                           'portable': nab('rds_cpu_utilization_e47b3b')}
scen['T2-CloudWatch-B'] = {'arm_bound': nab('ec2_cpu_utilization_5f5533'),
                           'x86_bound': nab('ec2_cpu_utilization_825cc2'),
                           'portable': nab('ec2_cpu_utilization_77c1ca')}
scen['T3-Diurnal-Taxi'] = {'arm_bound': taxi_window('2014-09-01'),
                           'x86_bound': taxi_window('2014-10-06'),
                           'portable': taxi_window('2014-12-01')}

t = np.arange(N); day = 288
def diurnal(phase, amp=0.5):
    week = np.where(((t // day) % 7) >= 5, 0.8, 1.0)
    return (1 + amp * np.sin(2 * np.pi * t / day - phase)) * week

def synth(kind, seed):
    out = {}
    for k, cls in enumerate(CLASSES):
        r = np.random.default_rng(seed * 10 + k)
        if kind == 'S1-Constant':
            v = 10 + r.normal(0, 0.5, N)
        elif kind == 'S2-Step':
            v = np.empty(N); i = 0
            while i < N:
                L = r.integers(72, 216); v[i:i+L] = r.uniform(5, 20); i += L
            v = v + r.normal(0, 0.5, N)
        elif kind == 'S3-Noisy':
            v = 10 * diurnal(k) * r.lognormal(0, 0.25, N)
            burst = r.random(N) < 0.02
            v[burst] *= r.uniform(1.5, 2.5, burst.sum())
        elif kind == 'S4-Batch':
            v = 2 + r.normal(0, 0.3, N); i = int(r.integers(0, 36))
            while i < N:
                L = int(r.integers(6, 19)); v[i:i+L] += r.uniform(15, 25); i += L + int(r.exponential(36))
        elif kind == 'S5-Spikes':
            v = 10 * diurnal(k, 0.3)
            small = r.random(N) < 0.3; v[small] *= 1 + r.uniform(0.10, 0.20, small.sum())
            for s in range(r.integers(0, 6), N, 6):      # large spike every ~30 min ...
                if r.random() < 0.15:                      # ... with prob 0.15 (cascading via shared seed below)
                    v[s:s+2] *= 1 + r.uniform(0.5, 0.7)
            v = v + r.normal(0, 0.3, N)
        elif kind == 'S6-Sinusoid':
            # manuscript workload lambda(t) = 20 + 15 sin(pi t / 60), t in minutes (2-h period)
            tm = t * 5.0
            v = (20 + 15 * np.sin(np.pi * tm / 60 + k)) / 35 * 12 + r.normal(0, 0.4, N)
        out[cls] = np.clip(v, 0.05, None)
    if kind == 'S5-Spikes':   # cascading: portable spikes propagate to bound classes 5 min later
        r = np.random.default_rng(seed + 99)
        idx = np.where(r.random(N) < 0.01)[0]
        for s in idx:
            out['portable'][s:s+2] *= 1.6
            for c in ('arm_bound', 'x86_bound'):
                out[c][s+1:s+3] *= 1.5
    return out

for i, kind in enumerate(['S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']):
    scen[kind] = synth(kind, 1000 + i)

for name, d in scen.items():
    df = pd.DataFrame({c: scale(d[c], c) for c in CLASSES})
    df.insert(0, 'minute', np.arange(N) * 5)
    df.to_csv(os.path.join(OUT, name + '.csv'), index=False)
    print(name, {c: (round(df[c].mean(), 2), round(df[c].max(), 2)) for c in CLASSES})
