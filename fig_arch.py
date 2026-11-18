import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
def box(x, y, w, h, title, body, fc='#f4f6f8', ec='#33475b', num=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', weight='bold')
    ax.text(x + w / 2, y + h - 6.2, body, ha='center', va='top', linespacing=1.35)
    if num: ax.text(x + 1.2, y + h - 1.0, num, ha='left', va='top', weight='bold', color='white',
                    bbox=dict(boxstyle='circle,pad=0.25', fc='#33475b', ec='none'))
def arrow(x1, y1, x2, y2, txt=None, dy=1.2, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, lw=1, color='#33475b', ls=ls))
    if txt: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha='center', va='bottom', fontsize=6.5, color='#33475b')
# cluster
box(1, 1, 30, 44, 'Kubernetes cluster', '', fc='#ffffff')
box(3, 26, 26, 12, 'ARM64 NodePool (Karpenter)', 'nodes: c6g.*  •  placeholder-arm64\nARM64-bound + portable pods', fc='#e8f1f8')
box(3, 11, 26, 12, 'x86-64 NodePool (Karpenter)', 'nodes: c5.*  •  placeholder-x86\nx86-64-bound + portable pods', fc='#fbeee9')
ax.text(16, 6.5, 'Karpenter: launches nodes for pending pods,\nconsolidates under-used nodes (fallback path)', ha='center', va='center', fontsize=6.8, style='italic')
box(35, 26, 17, 18, 'Metrics', 'Prometheus:\nCPU requests per\ndemand class\n(arm, x86, port)', num='1')
box(55, 26, 19, 18, 'Forecaster', 'multi-quantile LSTM\nper class, pinball loss\n$\\tau^*=\\rho/(1+\\rho)$\nhorizon 15 min', num='2')
box(77, 26, 22, 18, 'Planner', 'bound demand → own ISA\nportable demand → pool\nwith min $\\kappa_a p_a$\n$N_a=\\lceil T_a/C_{usable}\\rceil$', num='3')
box(48, 3, 40, 16, 'Actuator', 'set replicas of placeholder-<arch> Deployments\n(pause pods, negative PriorityClass);\nreal pods preempt placeholders on arrival', num='4')
arrow(31, 35, 35, 35)
arrow(52, 35, 55, 35); arrow(74, 35, 77, 35)
arrow(88, 26, 80, 19.5)
ax.text(91, 21.5, '$N_{arm}, N_{x86}$', fontsize=6.8, color='#33475b')
arrow(48, 11, 31, 17)
ax.text(37, 10.5, 'replicas', fontsize=6.8, color='#33475b')
fig.tight_layout(); fig.savefig('figs/architecture_v2.pdf'); fig.savefig('figs/architecture_v2.png', dpi=160)
