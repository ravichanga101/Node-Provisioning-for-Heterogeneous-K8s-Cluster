"""Statistical / simple forecasting baselines evaluated on the same rolling origins as the LSTMs."""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import logging; logging.getLogger('cmdstanpy').setLevel(logging.ERROR); logging.getLogger('prophet').setLevel(logging.ERROR)
W, H = 60, 6
scn = sys.argv[1]; D = 'data'; O = 'out'
df = pd.read_csv(os.path.join(D, scn + '.csv')); n = len(df); a = int(n * .70); b = int(n * .85)
o0, o1 = a - 1, n - H
res = {}
for s in ('arm_bound', 'x86_bound', 'portable'):
    y = df[s].values
    origins = np.arange(o0, o1)
    # naive (last value) and seasonal naive (same time previous day)
    res[f'{s}|naive|0'] = np.repeat(y[origins][:, None], H, 1)
    res[f'{s}|snaive|0'] = np.stack([y[o + 1 - 288:o + 1 - 288 + H] for o in origins])
    # ridge linear model on the 60-step window (DLinear-style direct multi-output)
    from sklearn.linear_model import Ridge
    X = np.stack([y[o - W + 1:o + 1] for o in range(W - 1, a - H)]); Y = np.stack([y[o + 1:o + 1 + H] for o in range(W - 1, a - H)])
    m = Ridge(alpha=1.0).fit(X, Y)
    res[f'{s}|ridge|0'] = m.predict(np.stack([y[o - W + 1:o + 1] for o in origins]))
    # ARIMA: order by AIC on training split, then rolling forecasts with fixed parameters
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    for order in [(1,0,0),(2,0,0),(1,0,1),(2,0,1),(1,1,1),(2,1,1),(0,1,1)]:
        try:
            r = ARIMA(y[:a], order=order).fit()
            if best is None or r.aic < best[0]: best = (r.aic, order, r)
        except Exception: pass
    rf = best[2].apply(y[:o1 + H])
    P = np.zeros((len(origins), H))
    for i, o in enumerate(origins):
        P[i] = rf.get_prediction(start=o + 1, end=o + H, dynamic=True).predicted_mean
    res[f'{s}|arima|0'] = P
    res.setdefault('_orders', {})[s] = str(best[1])
    # Prophet: refit every 24 h on all history; daily + weekly seasonality
    from prophet import Prophet
    ts = pd.date_range('2025-03-01', periods=n, freq='5min')
    P = np.zeros((len(origins), H)); last = None
    for i, o in enumerate(origins):
        if last is None or o - last >= 288:
            mdl = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            mdl.fit(pd.DataFrame({'ds': ts[:o + 1], 'y': y[:o + 1]}))
            fut = pd.DataFrame({'ds': ts[o + 1:min(n, o + 1 + 288 + H)]})
            yh = mdl.predict(fut).yhat.values; last = o; base = o
        P[i] = yh[o - base:o - base + H]
    res[f'{s}|prophet|0'] = P
    print(scn, s, 'done', best[1], flush=True)
orders = res.pop('_orders')
np.savez_compressed(os.path.join(O, scn + '_base.npz'), **res)
json.dump(orders, open(os.path.join(O, scn + '_arima_orders.json'), 'w'))
