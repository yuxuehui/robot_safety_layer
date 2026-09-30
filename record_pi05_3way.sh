#!/bin/bash
# pi0.5 three-way demo recordings (none | inference-time guidance | guidance + CAR vector) with the CURRENT code and the
# exact benchmark flags of obs7_jobs.sh (arms none / cbfjac_relesc / carvecjac_relesc), gate17 layout, libero_spatial
# task:episode EPS (default 0:3), torch seeds 0-2.  Usage: LAYOUT=gate20 EPS=2:3 TAG=gate20_t2e3 bash record_pi05_3way.sh [arms: none sota car]
# -> runs/video_3way/<TAG>_<arm>
source /workspace/mnt/mywang87/0Xuehui/oc_guidance_libero/env.sh
export PYTHONPATH=$PYTHONPATH:$PROJ OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
C=$PROJ/runs/baseline/libero_spatial/action_model_calibration.npz
CBF="--cost_type cbf --margin_grip 0.010 --margin_arm 0.015"
G1="--guidance g1 --scale 1.0 --grad_through_model 0"
LAYOUT=${LAYOUT:-gate17}
case $LAYOUT in gate17) LAY="--layout gate --gate_half_gap 0.17 --clean_gate 0.075" ;; gate20) LAY="--layout gate --gate_half_gap 0.20 --clean_gate 0.075" ;; single) LAY="--layout single" ;; esac
EPS=${EPS:-0:3}; TAG=${TAG:-gate17}
cd $PROJ/openpi
g=3
for arm in ${@:-none sota car}; do
  case $arm in
    none) EX="--guidance none" ;;
    sota) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
    car)  EX="--guidance car --scale 1.0 --car_conflict both --car_reward progress --car_zero_thr 0.01 --car_batch 32 --car_train_steps 1 --car_explore 0.03 --car_param vector --car_vec_beta 0.5 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  OUT=$PROJ/runs/video_3way/${TAG}_$arm; mkdir -p $OUT/logs
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python ../eval_obstacle.py --suite libero_spatial --eps $EPS --torch_seed $s --record \
        --calib $C --out $OUT $LAY $EX > $OUT/logs/seed$s.log 2>&1 < /dev/null &
    g=$(( (g+1) % 4 )); sleep 3
  done
done
echo "launched $(date)"
