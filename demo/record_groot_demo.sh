#!/bin/bash
# GR00T N1.7 demo recordings for the deck: per layout one libero_spatial episode whose benchmark outcome was
# "no guidance collides in all 3 seeds, guided is safe in all 3 seeds", recorded for none / ours x seeds 0-2
# (30 short runs against the running groot_servers.sh servers). Output: runs/video_groot/<layout>_<none|ours>_s<seed>.
source /workspace/mnt/mywang87/0Xuehui/oc_guidance_libero/env.sh
export PYTHONPATH=$PYTHONPATH:$PROJ OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
CG=$PROJ/runs/baseline_groot/libero_spatial/action_model_calibration.npz
G1="--guidance g1 --scale 1.0 --grad_through_model 0"; CBF="--cost_type cbf --margin_grip 0.010 --margin_arm 0.015"
PIL="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4"
HND="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1"
declare -A LAY=([single]="--layout single" [gate17]="--layout gate --gate_half_gap 0.17 --clean_gate 0.075"
                [gate20]="--layout gate --gate_half_gap 0.20 --clean_gate 0.075"
                [hsweep]="--extra_steps 100 --layout hand --hand_motion sweep --hand_visible 1"
                [hreach]="--extra_steps 100 --layout hand --hand_motion reach --hand_visible 1")
# chosen from the benchmark results (groot_<layout>_{none,sota}_s{0,1,2}): task:episode of libero_spatial
declare -A EPS=([single]="0:3" [gate17]="0:3" [gate20]="7:2" [hsweep]="1:3" [hreach]="8:2")
cd $PROJ/openpi
g=0
for L in single gate17 gate20 hsweep hreach; do
  for arm in none ours; do
    if [ $arm = none ]; then EX="--guidance none"; port=5562; else port=5561; case $L in h*) EX="$HND";; *) EX="$PIL";; esac; fi
    for s in 0 1 2; do
      name=video_groot/${L}_${arm}_s$s; mkdir -p $PROJ/runs/$name
      CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python ../benchmarks/eval_obstacle.py --suite libero_spatial --eps ${EPS[$L]} \
          --policy groot --groot_port $port --replan_steps 8 --baseline_dir ../runs/baseline_groot --calib $CG \
          --out ../runs/$name --torch_seed $s --record ${LAY[$L]} $EX > $PROJ/runs/$name/run.log 2>&1 < /dev/null &
      g=$(( (g+1) % 4 )); sleep 1
    done
  done
done
sleep 30
while [ $(ps -eo args | grep "[e]val_obstacle" | grep -c video_groot) -gt 0 ]; do sleep 20; done
echo "recordings done $(date)"
for L in single gate17 gate20 hsweep hreach; do
  for arm in none ours; do
    echo -n "$L $arm: "
    for s in 0 1 2; do
      r=$(grep "^{" $PROJ/runs/video_groot/${L}_${arm}_s$s/run.log | tail -1)
      python3 -c "import json,sys; r=json.loads(sys.argv[1]); print(('skip' if 'skipped' in r else ('S' if r['safe_success'] else ('C' if r['collided_any'] else 'x'))) + str(r.get('steps','')), end='  ')" "$r" 2>/dev/null || echo -n "?  "
    done; echo
  done
done
