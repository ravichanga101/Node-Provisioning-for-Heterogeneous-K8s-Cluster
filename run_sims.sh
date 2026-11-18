#!/bin/bash
cd /home/claude/exp
for e in main tau horizon boot kappa scale bootH; do
  [ -f out/sim_$e.csv ] || NPROC=2 python3 -I experiments.py $e >> out/sims.log 2>&1
done
echo SIMSDONE >> out/sims.log
