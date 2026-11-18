"""Train/evaluate forecasters for one scenario. Saves per-origin multi-step forecasts (vCPU)."""
import os, sys, time, json, argparse, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
W, H = 60, 6                      # 60 x 5-min look-back (5 h); 6 x 5-min horizon (30 min)
QS = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
COST = {'arm': 0.034, 'x86': 0.0425}   # USD per vCPU-hour (c6g.xlarge / c5.xlarge on-demand, us-east-1)

def split_idx(n):
    a = int(n * 0.70); b = int(n * 0.85); return a, b

def windows(z, start, end):
    """origins o in [start, end): input z[o-W+1..o], target z[o+1..o+H], both expressed relative to the
    last observation z[o] (window-relative normalization; robust to level shifts outside the training range)"""
    last = np.array([z[o] for o in range(start, end)])
    X = np.stack([z[o - W + 1:o + 1] - z[o] for o in range(start, end)])[..., None]
    Y = np.stack([z[o + 1:o + 1 + H] - z[o] for o in range(start, end)])
    return X.astype('float32'), Y.astype('float32')

def last_values(z, start, end):
    return np.array([z[o] for o in range(start, end)])

def build(kind, seed, cost=None):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    out = H * len(QS) if kind == 'quantile' else H
    m = tf.keras.Sequential([tf.keras.Input((W, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=True), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(out)])
    if kind == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError()
    elif kind == 'hybrid':          # initial-submission loss: lambda*MAE + (1-lambda)*MAE*Cost*5/60
        lam, c = 0.7, cost
        loss = lambda y, p: lam * tf.reduce_mean(tf.abs(p - y)) + (1 - lam) * tf.reduce_mean(tf.abs(p - y)) * c * 5 / 60
    else:
        q = tf.constant(np.repeat(QS[None, :], H, 0).reshape(-1), tf.float32)
        def loss(y, p):
            yy = tf.repeat(y, len(QS), axis=1)          # [b, H*Q] ordered h-major
            e = yy - p
            return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss=loss)
    return m

def train_lstm(z, a, b, kind, seed, cost=None):
    import tensorflow as tf
    Xtr, Ytr = windows(z, W - 1, a - H); Xva, Yva = windows(z, a - 1, b - H)
    m = build(kind, seed, cost)
    es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    t0 = time.time()
    h = m.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=40, batch_size=256, verbose=0, callbacks=[es], shuffle=True)
    return m, time.time() - t0, len(h.history['loss'])

def predict(m, z, start, end, kind):
    X, _ = windows(z, start, end)
    p = m.predict(X, batch_size=512, verbose=0)
    if kind == 'quantile':
        p = np.sort(p.reshape(len(p), H, len(QS)), axis=2)   # enforce non-crossing
    return p

def run_one(scn, s, kind, seed, data='data', out='out'):
    """Train one model and save its per-origin forecasts (resumable)."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    os.makedirs(os.path.join(out, 'pred'), exist_ok=True)
    fn = os.path.join(out, 'pred', f'{scn}__{s}__{kind}__{seed}.npy')
    if os.path.exists(fn): return
    df = pd.read_csv(os.path.join(data, scn + '.csv'))
    if s.endswith('_pool_v1'):
        y = df[s.split('_')[0] + '_bound'].values + 0.5 * df.portable.values
    else:
        y = df[s].values
    n = len(df); a, b = split_idx(n)
    lo, hi = y[:a].min(), y[:a].max(); sc = hi - lo; z = (y - lo) / sc     # scale fitted on training split only
    k = kind.split('_')[0]; cost = COST.get(kind.split('_')[1]) if kind.startswith('hybrid') else None
    m, dur, ep = train_lstm(z, a, b, k, seed, cost)
    p = predict(m, z, a - 1, n - H, 'quantile' if k == 'quantile' else 'mae')
    lv = last_values(z, a - 1, n - H)
    p = (p + (lv[:, None, None] if p.ndim == 3 else lv[:, None])) * sc + lo
    np.save(fn + '.tmp.npy', p); os.replace(fn + '.tmp.npy', fn)
    with open(os.path.join(out, 'pred', 'train_log.csv'), 'a') as f:
        f.write(f'{scn},{s},{kind},{seed},{dur:.1f},{ep},{a},{b},{n}\n')
    print(scn, s, kind, seed, f'{dur:.0f}s', ep, flush=True)

if __name__ == '__main__':
    run_one(*sys.argv[1:4], int(sys.argv[4]))
