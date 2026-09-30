#!/bin/bash
# obs7: clean-gate layouts (+ held-object model), 3 torch seeds. Writes one job line per (condition, shard) to
# stdout; run them with queue_run.sh.  Usage: bash obs7_jobs.sh <layout: gate17|gate20> <seeds: "0 1 2"> [arms...]
L=$1; SEEDS=$2; shift 2
PROJ=/workspace/mnt/mywang87/0Xuehui/oc_guidance_libero
C=$PROJ/runs/baseline/libero_spatial/action_model_calibration.npz
case $L in
  gate17) LAY="--layout gate --gate_half_gap 0.17 --clean_gate 0.075" ;;
  gate20) LAY="--layout gate --gate_half_gap 0.20 --clean_gate 0.075" ;;
  single) LAY="--layout single" ;;
  hsweepvis) LAY="--extra_steps 100 --layout hand --hand_motion sweep --hand_visible 1" ;;
  hsweepinv) LAY="--extra_steps 100 --layout hand --hand_motion sweep --hand_visible 0" ;;
  hreachvis) LAY="--extra_steps 100 --layout hand --hand_motion reach --hand_visible 1" ;;
  hreachinv) LAY="--extra_steps 100 --layout hand --hand_motion reach --hand_visible 0" ;;
  *) echo "unknown layout $L" >&2; exit 1 ;;
esac
CBF="--cost_type cbf --margin_grip 0.010 --margin_arm 0.015"
G1="--guidance g1 --scale 1.0 --grad_through_model 0"
ARMS=${@:-none cbf cbfobj steerobj steerexecobj mppiobj}
for seed in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      none) EX="--guidance none" ;;
      cbf) EX="$G1 $CBF" ;;                                   # ablation: no held-object model
      cbfobj) EX="$G1 $CBF --model_object 1" ;;
      cbfobj_rep) EX="$G1 $CBF --model_object 1" ;;           # same-seed replicate: noise floor
      steerobj) EX="$G1 $CBF --model_object 1 --steer" ;;
      steerexecobj) EX="$G1 $CBF --model_object 1 --steer --trigger_exec_only" ;;
      mppiobj) EX="--guidance mppi --mppi_w_ctrl 0.05 --mppi_w_dev 0 --mppi_w_obs 20 $CBF --model_object 1" ;;
      mppipers1) EX="--guidance mppi --mppi_w_ctrl 0.05 --mppi_w_dev 0 --mppi_w_obs 20 --mppi_persist 1.0 $CBF --model_object 1" ;;
      mppipers5) EX="--guidance mppi --mppi_w_ctrl 0.05 --mppi_w_dev 0 --mppi_w_obs 20 --mppi_persist 0.5 $CBF --model_object 1" ;;
      carvecobj) EX="--guidance car --scale 1.0 --car_conflict both --car_reward progress --car_zero_thr 0.01 --car_batch 32 --car_train_steps 1 --car_explore 0.03 --car_param vector --car_vec_beta 0.5 $CBF --model_object 1" ;;
      cbfjac) EX="$G1 $CBF --model_object 1 --arm_motion jac" ;;
      mppijac) EX="--guidance mppi --mppi_w_ctrl 0.05 --mppi_w_dev 0 --mppi_w_obs 20 $CBF --model_object 1 --arm_motion jac" ;;
      steerexecjac) EX="$G1 $CBF --model_object 1 --steer --trigger_exec_only --arm_motion jac" ;;
      mppijac_relesc) EX="--guidance mppi --mppi_w_ctrl 0.05 --mppi_w_dev 0 --mppi_w_obs 20 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
      cbfjac_rel) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1" ;;
      cbfjac_esc) EX="$G1 $CBF --model_object 1 --arm_motion jac --escape_chunks 4" ;;
      cbfjac_relesc) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
      steerexecjac_relesc) EX="$G1 $CBF --model_object 1 --steer --trigger_exec_only --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
      carvecjac_relesc) EX="--guidance car --scale 1.0 --car_conflict both --car_reward progress --car_zero_thr 0.01 --car_batch 32 --car_train_steps 1 --car_explore 0.03 --car_param vector --car_vec_beta 0.5 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
      sota_cv) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.03 --hand_margin_tau 0.2" ;;
      sota_or) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict oracle --margin_hand 0.03 --hand_margin_tau 0.2" ;;
      sota_st) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict static --margin_hand 0.03 --hand_margin_tau 0.2" ;;
      sota_cv_t) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1" ;;
      sota_or_t) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict oracle --margin_hand 0.0 --hand_margin_tau 0.1" ;;
      none_r2) EX="--guidance none --replan_steps 2" ;;
      sota_cv_t_r2) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1 --replan_steps 2" ;;
      sota_cv_r2) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.03 --hand_margin_tau 0.2 --replan_steps 2" ;;
      none_r8) EX="--guidance none --replan_steps 8" ;;
      sota_r2) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4 --replan_steps 2" ;;
      sota_r8) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4 --replan_steps 8" ;;
      sota_cv_esc) EX="$G1 $CBF --model_object 1 --arm_motion jac --escape_chunks 4 --hand_predict cv --margin_hand 0.03 --hand_margin_tau 0.2" ;;
      carobj) EX="--guidance car --scale 1.0 --car_zero_thr 0.01 $CBF --model_object 1" ;;
      car2obj) EX="--guidance car --scale 1.0 --car_conflict both --car_reward progress --car_zero_thr 0.01 --car_batch 32 --car_train_steps 4 --car_lr 1e-2 --car_explore 0.03 $CBF --model_object 1" ;;
      *) echo "unknown arm $arm" >&2; continue ;;
    esac
    name=obs7_${L}_${arm}_s$seed
    for shard in "libero_spatial 0-3" "libero_spatial 4-6" "libero_spatial 7-9" "libero_object 0-3" "libero_object 4-6" "libero_object 7-9"; do
      suite=${shard%% *}; tasks=${shard#* }
      echo "$name|$suite|$tasks|--suite $suite --tasks $tasks --episodes ${EPISODES:-10} --calib $C --out ../runs/$name --torch_seed $seed $LAY $EX"
    done
  done
done
