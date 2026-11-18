"""Proposition 1 check with deterministic TensorFlow ops: MAE vs hybrid loss, same seed, same data order."""
import os, json, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as F
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
scn, c = 'T3-Diurnal-Taxi', 'arm_bound'
df = pd.read_csv(f'data/{scn}.csv'); y = df[c].values; n = len(y); a, b = F.split_idx(n)
lo, hi = y[:a].min(), y[:a].max(); sc = hi - lo; z = (y - lo) / sc
def run(kind, seed, cost=None):
    m, dur, ep = F.train_lstm(z, a, b, kind, seed, cost)
    p = F.predict(m, z, b - 1, n - F.H, 'mae'); lv = F.last_values(z, b - 1, n - F.H)
    return (p + lv[:, None]) * sc + lo, ep
Y = np.stack([y[o + 1:o + 1 + F.H] for o in range(b - 1, n - F.H)])
res = {}
P0, e0 = run('mae', 0); P0b, _ = run('mae', 0)
Ph_arm, e1 = run('hybrid', 0, 0.034); Ph_x86, e2 = run('hybrid', 0, 0.0425); P1, _ = run('mae', 1)
mae = lambda P: float(np.abs(P[:, :3] - Y[:, :3]).mean())
res = dict(MAE_mae_seed0=mae(P0), MAE_mae_seed0_repeat=mae(P0b), MAE_hybrid_arm=mae(Ph_arm), MAE_hybrid_x86=mae(Ph_x86), MAE_mae_seed1=mae(P1),
           maxdiff_repeat=float(np.abs(P0 - P0b).max()), maxdiff_hybrid_arm=float(np.abs(P0 - Ph_arm).max()),
           maxdiff_hybrid_x86=float(np.abs(P0 - Ph_x86).max()), maxdiff_seed1=float(np.abs(P0 - P1).max()),
           meandiff_hybrid_arm=float(np.abs(P0 - Ph_arm).mean()), meandiff_seed1=float(np.abs(P0 - P1).mean()), epochs=[e0, e1, e2])
print(res); json.dump(res, open('out/hybrid_check.json', 'w'), indent=1)
