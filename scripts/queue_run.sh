#!/bin/bash
# Run eval_obstacle jobs from a job file with at most K concurrent processes, GPUs round-robin.
# Usage: setsid nohup bash scripts/queue_run.sh <jobfile> <K> [gpu0] > queue.log 2>&1 &
# Job line format (from scripts/obs7_jobs.sh): name|suite|tasks|<eval args>
JOBS=$1; K=$2; gpu=${3:-0}
cd /workspace/mnt/mywang87/0Xuehui/oc_guidance_libero
source env.sh
export PYTHONPATH=$PYTHONPATH:$PROJ OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
NGPU=$(nvidia-smi -L | wc -l)
cd openpi
while IFS='|' read -r name suite tasks eargs; do
  [ -z "$name" ] && continue
  while [ "$(pgrep -c -f 'python ../benchmarks/eval_obstacle.py --suite')" -ge "$K" ]; do sleep 30; done
  # GPU with the fewest eval processes (CUDA_VISIBLE_DEVICES of running eval_obstacle processes; new ones count too)
  gpu=$(for g in $(seq 0 $((NGPU-1))); do echo "$(ps -eo pid,args | grep "[e]val_obstacle.py --suite" | awk '{print $1}' | while read p; do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -c "^CUDA_VISIBLE_DEVICES=$g$"; done | awk '{s+=$1} END {print s+0}') $g"; done | sort -n | head -1 | cut -d" " -f2)
  mkdir -p $PROJ/runs/$name/logs
  echo "$(date +%H:%M:%S) start $name $suite $tasks gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup .venv/bin/python ../benchmarks/eval_obstacle.py $eargs \
    > $PROJ/runs/$name/logs/${suite}_${tasks}.log 2>&1 < /dev/null &
  sleep 30
done < "$JOBS"
echo "$(date +%H:%M:%S) all jobs launched"
