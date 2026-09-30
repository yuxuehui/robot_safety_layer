#!/bin/bash
# GR00T N1.7 obstacle-free baselines (all tasks x 10 episodes of libero_spatial and libero_object) through the running
# servers (groot_servers.sh), then the action -> EEF gain calibration on them. Output: runs/baseline_groot/<suite>/.
# Usage: bash groot_baseline.sh [episodes=10]
EPS=${1:-10}
source /workspace/mnt/mywang87/0Xuehui/oc_guidance_libero/env.sh
export PYTHONPATH=$PYTHONPATH:$PROJ OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu
cd $PROJ/openpi
mkdir -p $PROJ/runs/baseline_groot
run() { gpu=$1; port=$2; suite=$3; tasks=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup .venv/bin/python ../benchmarks/eval_libero.py --policy groot --groot_port $port --suite $suite --tasks $tasks \
      --episodes $EPS --replan_steps 8 --out ../runs/baseline_groot > $PROJ/runs/baseline_groot/${suite}_${tasks}.log 2>&1 < /dev/null &
}
# two client processes per server: the zmq REP socket serialises them, the sim/render work overlaps
run 1 5562 libero_spatial 0-4;  run 1 5562 libero_spatial 5-9
run 3 5564 libero_object 0-4;   run 3 5564 libero_object 5-9
echo "launched 4 baseline shards $(date); wait with: until [ \$(pgrep -fc 'eval_liber[o].py --policy groot') -eq 0 ]; do sleep 60; done"
wait_all() { until [ $(pgrep -fc "eval_liber[o].py --policy groot") -eq 0 ]; do sleep 60; done; }
wait_all
echo "baselines done $(date)"
for suite in libero_spatial libero_object; do
  echo "== $suite: $(cat $PROJ/runs/baseline_groot/$suite/results_*.jsonl | wc -l) episodes, success $(cat $PROJ/runs/baseline_groot/$suite/results_*.jsonl | grep -c '"success": true')"
  .venv/bin/python ../benchmarks/calibrate_action_model.py ../runs/baseline_groot/$suite 2>&1 | tail -8
done
