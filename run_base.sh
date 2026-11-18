#!/bin/bash
cd /home/claude/exp
for s in T1-CloudWatch-A T2-CloudWatch-B T3-Diurnal-Taxi S1-Constant S2-Step S3-Noisy S4-Batch S5-Spikes S6-Sinusoid; do
  [ -f out/${s}_base.npz ] || nice -n 19 python3 -I baselines.py $s >> out/base.log 2>&1
done
echo BASEDONE >> out/base.log
