"""Trace-driven discrete-time simulator of a heterogeneous (ARM64 / x86-64) Kubernetes node pool
autoscaled by Karpenter-style provisioning, with reactive, threshold and proactive controllers."""
import numpy as np, math

DT = 30.0                       # simulation step (s)
NODE_VCPU = 4.0                 # c6g.xlarge / c5.xlarge
NODE_USABLE = 3.5               # allocatable after system reservation, at 0.5-vCPU pod granularity
PRICE = {'arm': 0.034, 'x86': 0.0425}   # USD per vCPU-hour, on-demand us-east-1
SVC_S = 0.020                   # mean service time of the latency-sensitive service (s)
SLA_P95 = 0.250                 # p95 response-time objective (s)
USAGE_RATIO = 0.70              # CPU usage / CPU request of running pods (HPA target utilisation)
POOLS = ('arm', 'x86')

def erlang_c_p95(A, c):
    """p95 response time of an M/M/c queue with offered load A (Erlang) and c servers."""
    if c <= 0: return math.inf
    if A >= c: return math.inf
    B = 1.0
    for k in range(1, c + 1):
        B = A * B / (k + A * B)
    C = c * B / (c - A * (1 - B))
    mu = 1.0 / SVC_S
    w95 = max(0.0, math.log(C / 0.05) / ((c - A) * mu)) if C > 0.05 else 0.0
    return SVC_S + w95

class Node:
    __slots__ = ('pool', 't_launch', 't_ready', 't_end')
    def __init__(s, pool, t, boot):
        s.pool = pool; s.t_launch = t; s.t_ready = t + boot; s.t_end = None

def run(demand, fc, cfg, seed=0, record=False):
    """demand: dict class -> array at 5-min resolution covering [origin0 .. end];
    fc: dict with forecast arrays aligned so that fc[key][j] is the forecast made at data index j (latest sample j)
    cfg: strategy configuration."""
    rng = np.random.default_rng(seed)
    nT = len(demand['arm_bound']) - 1           # simulate between first and last data point
    steps = int(nT * 300 / DT)
    kappa = cfg.get('kappa', 1.0)               # ARM vCPU needed per unit of portable demand
    eff_price = {'arm': PRICE['arm'] * kappa, 'x86': PRICE['x86']}
    cheap = min(POOLS, key=lambda p: eff_price[p])
    boot_med = cfg.get('boot_median', 60.0)
    nmin, nmax = cfg.get('nmin', 1), cfg.get('nmax', 12)
    nodes = []
    def boot():
        return float(boot_med * rng.lognormal(0, 0.3))
    # initial cluster: enough ready nodes for the first sample at 70% target
    d0 = {c: demand[c][0] for c in demand}
    init = {'arm': d0['arm_bound'] + kappa * d0['portable'] / 2, 'x86': d0['x86_bound'] + d0['portable'] / 2}
    for p in POOLS:
        for _ in range(max(nmin, math.ceil(init[p] / (0.7 * NODE_USABLE)))):
            n = Node(p, -600.0, 0.0); n.t_ready = -1.0; nodes.append(n)
    noise = {c: rng.lognormal(0, 0.05, steps) for c in demand}
    protect = {p: 0 for p in POOLS}; idle_since = {p: None for p in POOLS}
    last_action = {p: -1e9 for p in POOLS}
    M = dict(pending=np.zeros(steps), viol=np.zeros(steps, bool), short=np.zeros(steps, bool), served=0.0, cap_ready=0.0,
             util_pool={p: [0.0, 0.0] for p in POOLS}, episodes=[], scale_out=0, absorbed=0)
    ep_start = {p: None for p in POOLS}
    rec = {k: [] for k in ('t', 'd_arm', 'd_x86', 'cap_arm', 'cap_x86', 'pend')} if record else None
    portable_split = cfg.get('portable', 'cheapest')
    from collections import deque
    peak_hist = {c: deque(maxlen=288) for c in demand}      # last 24 h of per-interval peak/sample ratios
    cur_peak = {c: 1.0 for c in demand}
    spi = int(300 / DT)
    for k in range(steps):
        t = k * DT
        x = t / 300.0; i = int(x); f = x - i
        base = {c: demand[c][i] * (1 - f) + demand[c][min(i + 1, nT)] * f for c in demand}
        d = {c: base[c] * noise[c][k] for c in demand}
        if k % spi == 0 and k > 0:
            for c in demand: peak_hist[c].append(cur_peak[c]); cur_peak[c] = 0.0
        for c in demand: cur_peak[c] = max(cur_peak[c], d[c] / max(base[c], 1e-9))
        alive = [n for n in nodes if n.t_end is None]
        ready = {p: sum(1 for n in alive if n.pool == p and n.t_ready <= t) for p in POOLS}
        launching = {p: sum(1 for n in alive if n.pool == p and n.t_ready > t) for p in POOLS}
        R = {p: ready[p] * NODE_USABLE for p in POOLS}
        # --- allocate demand ---
        bound = {'arm': d['arm_bound'], 'x86': d['x86_bound']}
        free = {p: R[p] - bound[p] for p in POOLS}
        pend = {p: max(0.0, -free[p]) for p in POOLS}
        served = {p: min(bound[p], R[p]) for p in POOLS}
        port = d['portable']
        if portable_split == 'cheapest':
            pref = [(cheap, port)]
        else:
            pref = [('arm', port / 2), ('x86', port / 2)]
        for p, amt in pref:
            need = amt * (kappa if p == 'arm' else 1.0)
            use = min(max(free[p], 0.0), need)
            served[p] += use; free[p] -= use; rem = need - use
            if rem > 1e-9:          # spill to the other pool's free capacity (scheduler places anywhere)
                o = 'x86' if p == 'arm' else 'arm'
                rem_units = rem / (kappa if p == 'arm' else 1.0) * (kappa if o == 'arm' else 1.0)
                use2 = min(max(free[o], 0.0), rem_units)
                served[o] += use2; free[o] -= use2
                left = (rem_units - use2) / (kappa if o == 'arm' else 1.0) * (kappa if p == 'arm' else 1.0)
                pend[p] += left       # pending portable pods target the preferred pool
        # --- proactive planner at 5-minute boundaries ---
        if cfg['kind'] in ('v1', 'v2') and k % int(300 / DT) == 0:
            j = i
            targ = plan(cfg, fc, j, demand, kappa, cheap, nmax, R, bound, served, pend, peak_hist)
            if targ is not None:
                for p in POOLS:
                    want = targ[p]
                    have = ready[p] + launching[p]
                    if cfg['kind'] == 'v1':
                        add = want            # v1 returns nodes to ADD for a single architecture
                    else:
                        add = want - have
                        protect[p] = min(want, nmax)
                    for _ in range(int(max(0, min(add, nmax - have)))):
                        nodes.append(Node(p, t, boot())); launching[p] += 1; M['scale_out'] += 1
        # --- threshold controller ---
        if cfg['kind'] == 'threshold':
            for p in POOLS:
                u = served[p] / R[p] if R[p] > 0 else 1.0
                have = ready[p] + launching[p]
                if (u > cfg.get('thr', 0.70) or pend[p] > 0) and t - last_action[p] >= cfg.get('cooldown', 300):
                    need = math.ceil((served[p] + pend[p]) / (cfg.get('thr', 0.70) * NODE_USABLE))
                    for _ in range(max(0, min(need - have, nmax - have))):
                        nodes.append(Node(p, t, boot())); launching[p] += 1; M['scale_out'] += 1
                    last_action[p] = t
        # --- reactive (Karpenter pending-pod) fallback ---
        if cfg.get('reactive', True) and cfg['kind'] != 'threshold':
            for p in POOLS:
                uncovered = pend[p] - launching[p] * NODE_USABLE
                if uncovered > 1e-6:
                    have = ready[p] + launching[p]
                    for _ in range(min(math.ceil(uncovered / NODE_USABLE), nmax - have)):
                        nodes.append(Node(p, t, boot())); launching[p] += 1; M['scale_out'] += 1
        # --- scale-in ---
        for p in POOLS:
            if cfg.get('scalein', 'consolidate') == 'idle10':      # legacy idle-node rule (not used in the paper)
                u_node = served[p] / (ready[p] * NODE_VCPU) if ready[p] else 1.0
                cond = u_node < 0.10
            elif cfg['kind'] == 'threshold':
                cond = (served[p] / R[p] if R[p] else 1) < 0.5 and pend[p] == 0
            else:
                cond = free[p] >= NODE_USABLE and ready[p] > protect[p] and pend[p] == 0
            if cond and ready[p] > nmin:
                if idle_since[p] is None: idle_since[p] = t
                elif t - idle_since[p] >= cfg.get('idle_after', 300):
                    cand = [n for n in nodes if n.t_end is None and n.pool == p and n.t_ready <= t]
                    cand[-1].t_end = t; idle_since[p] = None
                    if cfg['kind'] == 'threshold': last_action[p] = t
            else:
                idle_since[p] = None
        # --- metrics ---
        ptot = pend['arm'] + pend['x86']
        M['pending'][k] = ptot
        viol = False; short = ptot > 1e-6
        for p in POOLS:
            M['util_pool'][p][0] += served[p]; M['util_pool'][p][1] += ready[p] * NODE_VCPU
            # demand is expressed in CPU *requests*; pods use USAGE_RATIO of their request (HPA target).
            # Unscheduled replicas push their load onto the scheduled ones (M/M/c, c = scheduled vCPU).
            U = USAGE_RATIO * (served[p] + pend[p])
            r95 = erlang_c_p95(U, max(1, int(math.ceil(served[p] - 1e-9)))) if U > 0 else 0.0
            if r95 > SLA_P95: viol = True
            if pend[p] > 1e-6 and ep_start[p] is None: ep_start[p] = t
            if pend[p] <= 1e-6 and ep_start[p] is not None:
                M['episodes'].append(t - ep_start[p]); ep_start[p] = None
        M['viol'][k] = viol; M['short'][k] = short
        M['served'] += served['arm'] + served['x86']; M['cap_ready'] += (ready['arm'] + ready['x86']) * NODE_VCPU
        if record:
            rec['t'].append(t); rec['d_arm'].append(served['arm'] + pend['arm']); rec['d_x86'].append(served['x86'] + pend['x86'])
            rec['cap_arm'].append(ready['arm'] * NODE_USABLE); rec['cap_x86'].append(ready['x86'] * NODE_USABLE); rec['pend'].append(ptot)
    T_end = steps * DT
    cost = {p: 0.0 for p in POOLS}; node_h = {p: 0.0 for p in POOLS}
    for n in nodes:
        e = n.t_end if n.t_end is not None else T_end
        s = max(n.t_launch, 0.0)
        hrs = max(0.0, e - s) / 3600
        cost[n.pool] += hrs * NODE_VCPU * PRICE[n.pool]; node_h[n.pool] += hrs
    hours = T_end / 3600
    ep = np.array(M['episodes']) if M['episodes'] else np.array([0.0])
    out = dict(cost_per_day=(cost['arm'] + cost['x86']) / hours * 24,
               cost_arm_share=cost['arm'] / max(1e-9, cost['arm'] + cost['x86']),
               sla_viol_pct=100 * M['viol'].mean(),
               shortfall_pct=100 * M['short'].mean(),
               unserved_vcpu_min=M['pending'].sum() * DT / 60 / hours * 24,      # per day
               util_pct=100 * M['served'] / M['cap_ready'],
               util_arm=100 * M['util_pool']['arm'][0] / max(1e-9, M['util_pool']['arm'][1]),
               util_x86=100 * M['util_pool']['x86'][0] / max(1e-9, M['util_pool']['x86'][1]),
               pend_episodes_per_day=len(M['episodes']) / hours * 24,
               pend_ep_mean_s=float(ep.mean()), pend_ep_p95_s=float(np.percentile(ep, 95)),
               node_launches_per_day=M['scale_out'] / hours * 24,
               node_hours_per_day=(node_h['arm'] + node_h['x86']) / hours * 24)
    return (out, rec) if record else out

QS = [0.5, 0.8, 0.9, 0.95, 0.99]

def plan(cfg, fc, j, demand, kappa, cheap, nmax, R, bound, served, pend, peak_hist=None):
    h = cfg.get('h', 3)
    if cfg['kind'] == 'v1':
        # legacy single-architecture point-forecast planner (not used in the paper): pool-level point forecasts,
        # Creq = mean(3-step)/eta ; deficit = max(0, Creq + Dpend - Cfree); a* = argmin cost; FFD -> ceil
        eta = 0.7; deficits = {}
        for p in POOLS:
            D = fc[f'{p}_pool_v1'][j, :3].mean()
            Creq = D / eta
            Cfree = R[p] - served[p]
            deficits[p] = max(0.0, Creq + pend[p] - Cfree)
        cand = [p for p in POOLS if deficits[p] > 0]
        if not cand: return None
        a = min(cand, key=lambda p: deficits[p] * PRICE[p])
        return {p: (math.ceil(deficits[p] / NODE_USABLE) if p == a else 0) for p in POOLS}
    # ---------------- v2 planner ----------------
    if cfg.get('oracle'):
        fut = lambda c: max(demand[c][j + 1:j + 1 + h]) if j + 1 < len(demand[c]) else demand[c][j]
        q = {c: fut(c) for c in ('arm_bound', 'x86_bound', 'portable')}
    elif cfg.get('point'):
        eta = cfg.get('eta', 0.7)
        q = {c: fc[f'{c}|mae'][j, :3].mean() / eta for c in ('arm_bound', 'x86_bound', 'portable')}
    else:
        qi = cfg['qidx']
        q = {c: fc[f'{c}|q'][j, :h, qi].max() for c in ('arm_bound', 'x86_bound', 'portable')}
    q = {c: max(0.0, v) for c, v in q.items()}
    if cfg.get('burst', True) and peak_hist is not None:
        # intra-interval peak factor: tau*-quantile of the observed ratio between the 30-s peak and the
        # 5-min sample within each past interval (last 24 h); 1.0 until one hour of history exists
        tau = QS[cfg.get('qidx', 2)]
        for c in q:
            h = peak_hist[c]
            if len(h) >= 12: q[c] *= max(1.0, float(np.quantile(np.fromiter(h, float), tau)))
    T = {'arm': q['arm_bound'], 'x86': q['x86_bound']}
    if cfg.get('portable', 'cheapest') == 'cheapest':
        other = 'x86' if cheap == 'arm' else 'arm'
        unit = lambda p: kappa if p == 'arm' else 1.0
        room = nmax * NODE_USABLE - T[cheap]
        put = min(q['portable'] * unit(cheap), max(0.0, room))
        T[cheap] += put
        T[other] += (q['portable'] - put / unit(cheap)) * unit(other)
    else:
        T['arm'] += kappa * q['portable'] / 2; T['x86'] += q['portable'] / 2
    return {p: math.ceil(T[p] / NODE_USABLE) for p in POOLS}
