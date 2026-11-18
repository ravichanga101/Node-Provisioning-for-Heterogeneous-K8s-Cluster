"""Controller overhead: wall-clock time of one decision (3 quantile-LSTM inferences + planner), CPU only."""
import os, time, json, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as F, sim
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
models = [F.build('quantile', s) for s in range(3)]
x = np.random.rand(1, F.W, 1).astype('float32')
fns = [tf.function(m) for m in models]
for f in fns: f(x)                                     # warm-up / tracing
T = []
for _ in range(200):
    t0 = time.perf_counter()
    q = [f(x).numpy() for f in fns]
    T.append(time.perf_counter() - t0)
fc = {'arm_bound|q': np.random.rand(10, 6, 5), 'x86_bound|q': np.random.rand(10, 6, 5), 'portable|q': np.random.rand(10, 6, 5)}
P = []
for _ in range(2000):
    t0 = time.perf_counter(); sim.plan(dict(kind='v2', qidx=2), fc, 3, None, 1.0, 'arm', 12, None, None, None, None); P.append(time.perf_counter() - t0)
params = int(sum(m.count_params() for m in models) / 3)
res = dict(inference_ms_median=1e3 * np.median(T), inference_ms_p95=1e3 * np.percentile(T, 95), planner_us_median=1e6 * np.median(P),
           params_per_model=params, tf=tf.__version__, cpu=os.cpu_count())
print(res); json.dump(res, open('out/overhead.json', 'w'), indent=1)
