"""Run all simulation experiments; writes tidy CSVs to out/."""
import os, sys, json, time, itertools, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
SCN = ['T1-CloudWatch-A', 'T2-CloudWatch-B', 'T3-Diurnal-Taxi', 'S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']
QS = [0.5, 0.8, 0.9, 0.95, 0.99]
_cache = {}
SHORT = {'Reactive (Karpenter)': 'Reactive', 'Proposed v2': 'Proposed v2'}

def P(scn, s, kind, seed=0):
    return np.load(f'out/pred/{scn}__{s}__{kind}__{seed}.npy')

def load(scn, scale=1.0, fseed=0):
    key = (scn, scale, fseed)
    if key in _cache: return _cache[key]
    df = pd.read_csv(f'data/{scn}.csv'); n = len(df); a = int(n * .70); b = int(n * .85); H = 6
    o0 = a - 1; s0, s1 = b, n - H
    dem = {c: df[c].values[s0:s1] * scale for c in ('arm_bound', 'x86_bound', 'portable')}
    sl = slice(s0 - o0, s1 - o0)
    fc = {}
    for c in ('arm_bound', 'x86_bound', 'portable'):
        fc[f'{c}|mae'] = P(scn, c, 'mae', fseed)[sl] * scale
        fc[f'{c}|q'] = P(scn, c, 'quantile', fseed)[sl] * scale
    for p in ('arm', 'x86'):
        fc[f'{p}_pool_v1'] = P(scn, f'{p}_pool_v1', 'mae')[sl] * scale
    _cache[key] = (dem, fc); return dem, fc

STRAT = {
    'Reactive (Karpenter)': dict(kind='reactive'),
    'Threshold (70%, arch-agnostic)': dict(kind='threshold', portable='split'),
    'Proactive v1 (initial submission)': dict(kind='v1', portable='split', scalein='idle10'),
    'Proposed v2': dict(kind='v2', qidx=2),
    'v2 w/o quantile (point+eta)': dict(kind='v2', point=True),
    'v2 w/o portability': dict(kind='v2', qidx=2, portable='split'),
    'v2 w/o burst margin': dict(kind='v2', qidx=2, burst=False),
    'Oracle (perfect forecast)': dict(kind='v2', oracle=True),
}

def job(args):
    exp, scn, name, cfg, seed, scale = args
    dem, fc = load(scn, scale, cfg.get('fseed', 0))
    t0 = time.time(); r = sim.run(dem, fc, cfg, seed=seed); r['wall_s'] = time.time() - t0
    r.update(exp=exp, scenario=scn, strategy=name, seed=seed, scale=scale, **{f'cfg_{k}': v for k, v in cfg.items()})
    return r

def main(which):
    jobs = []
    if 'main' in which:
        for scn, (nm, cfg), sd in itertools.product(SCN, STRAT.items(), range(10)):
            jobs.append(('main', scn, nm, cfg, sd, 1.0))
    if 'tau' in which:
        for scn, qi, sd in itertools.product(SCN, range(5), range(5)):
            jobs.append(('tau', scn, f'tau={QS[qi]}', dict(kind='v2', qidx=qi), sd, 1.0))
    if 'horizon' in which:
        for scn, h, sd in itertools.product(SCN, (1, 2, 3, 6), range(5)):
            jobs.append(('horizon', scn, f'H={5*h}min', dict(kind='v2', qidx=2, h=h), sd, 1.0))
    if 'boot' in which:
        for scn, bm, (nm, cfg), sd in itertools.product(SCN, (60, 180, 420, 600),
                [(k, STRAT[k]) for k in ('Reactive (Karpenter)', 'Threshold (70%, arch-agnostic)', 'Proposed v2')], range(5)):
            c = dict(cfg, boot_median=bm); jobs.append(('boot', scn, nm, c, sd, 1.0))
    if 'kappa' in which:
        for scn, kp, (nm, cfg), sd in itertools.product(SCN, (0.8, 1.0, 1.25, 1.5),
                [(k, STRAT[k]) for k in ('Reactive (Karpenter)', 'Proposed v2', 'v2 w/o portability')], range(5)):
            c = dict(cfg, kappa=kp); jobs.append(('kappa', scn, nm, c, sd, 1.0))
    if 'bootH' in which:
        for scn, bm, sd in itertools.product(SCN, (420, 600), range(5)):
            jobs.append(('bootH', scn, 'Proposed v2 (H=30min)', dict(kind='v2', qidx=2, h=6, boot_median=bm), sd, 1.0))
    if 'scale' in which:
        for scn, (nm, cfg), sd in itertools.product(SCN, [(k, STRAT[k]) for k in ('Reactive (Karpenter)', 'Threshold (70%, arch-agnostic)', 'Proposed v2')], range(3)):
            jobs.append(('scale', scn, nm, dict(cfg, nmax=120), sd, 10.0))
    if 'extra' in which:
        for scn, sd in itertools.product(SCN, range(10)):
            jobs.append(('extra', scn, 'Threshold (70%, arch-aware)', dict(kind='threshold', portable='cheapest'), sd, 1.0))
            for fs in (1, 2):
                jobs.append(('extra', scn, f'Proposed v2 (model seed {fs})', dict(kind='v2', qidx=2, fseed=fs), sd, 1.0))
    with Pool(int(os.environ.get('NPROC', 2))) as p:
        rows = p.map(job, jobs, chunksize=4)
    df = pd.DataFrame(rows); tag = '_'.join(which)
    df.to_csv(f'out/sim_{tag}.csv', index=False); print('wrote', len(df), 'rows', tag)

if __name__ == '__main__':
    main(sys.argv[1:])
