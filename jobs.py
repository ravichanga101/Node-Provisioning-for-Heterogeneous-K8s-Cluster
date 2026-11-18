SCN = ['T1-CloudWatch-A', 'T2-CloudWatch-B', 'T3-Diurnal-Taxi', 'S1-Constant', 'S2-Step', 'S3-Noisy', 'S4-Batch', 'S5-Spikes', 'S6-Sinusoid']
CL = ['arm_bound', 'x86_bound', 'portable']
J = []
for scn in SCN:                               # seed 0: everything the simulator needs
    for c in CL: J += [(scn, c, 'quantile', 0), (scn, c, 'mae', 0)]
    J += [(scn, 'arm_pool_v1', 'mae', 0), (scn, 'x86_pool_v1', 'mae', 0)]
for c in ('arm_bound',):                      # loss-equivalence check (Proposition 1)
    J += [('T3-Diurnal-Taxi', c, 'hybrid_arm', 0), ('T3-Diurnal-Taxi', c, 'hybrid_x86', 0)]
for scn in SCN:                               # extra seeds for accuracy statistics (all workloads)
    for sd in (1, 2):
        for c in CL: J += [(scn, c, 'mae', sd), (scn, c, 'quantile', sd)]
import sys
q = int(sys.argv[1]); nq = int(sys.argv[2])
for i, j in enumerate(J):
    if i % nq == q: print(*j)
