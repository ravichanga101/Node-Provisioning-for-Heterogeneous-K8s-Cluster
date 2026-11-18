#!/bin/bash
cd /home/claude/exp
worker() { python3 -I jobs.py $1 2 | while read scn s kind seed; do python3 -I forecast.py $scn $s $kind $seed >> out/w$1.log 2>&1; done; }
worker 0 & worker 1 & wait
echo ALLDONE >> out/w0.log
