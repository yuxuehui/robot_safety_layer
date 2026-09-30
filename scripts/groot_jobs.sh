#!/bin/bash
# GR00T N1.7 version of obs7_jobs.sh: one job line per (condition, shard) for queue_run.sh. The policy runs behind the
# servers of groot_servers.sh (port by suite x guidance); pillars / hand are placed on GR00T's own baselines
# (runs/baseline_groot) and the gain is the one calibrated on them. Chunk: 16 valid steps, 8 executed (GR00T default).
# Usage: bash groot_jobs.sh <layout: gate17|gate20|single|hsweepvis|hsweepinv|hreachvis|hreachinv> <seeds: "0 1 2"> [arms: none sota]
L=$1; SEEDS=$2; shift 2
PROJ=/workspace/mnt/mywang87/0Xuehui/oc_guidance_libero
CG=$PROJ/runs/baseline_groot/libero_spatial/action_model_calibration.npz
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
ARMS=${@:-none sota}
for seed in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      none) EX="--guidance none"; mode=none ;;
      sota)  # inference-time guidance: pillars -> release + escape; hand -> cv prediction, tight margin, no escape
        case $L in
          h*) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1" ;;
          *)  EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;;
        esac; mode=g1 ;;
      # GR00T-specific ablation (2026-09-30): smaller push, earlier / looser deadlock escape, faster replanning
      sota_s05) case $L in h*) EX="--guidance g1 --scale 0.5 --grad_through_model 0 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1" ;;
                          *)  EX="--guidance g1 --scale 0.5 --grad_through_model 0 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4" ;; esac; mode=g1 ;;
      sota_esc2) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 2 --escape_min_disp 0.05"; mode=g1 ;;
      sota_r4) case $L in h*) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1 --replan_steps 4" ;;
                         *)  EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4 --replan_steps 4" ;; esac; mode=g1 ;;
      sota_gate2) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 6 --escape_min_disp 0.08 --escape_mode window --escape_ratio 2.0"; mode=g1 ;;
      sota_rec) case $L in h*) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1 --escape_chunks 6 --escape_min_disp 0.08 --escape_mode window --escape_ratio 2.0 --recovery 1" ;;
                         *)  EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 6 --escape_min_disp 0.08 --escape_mode window --escape_ratio 2.0 --recovery 1" ;; esac; mode=g1 ;;
      sota_la) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 6 --escape_min_disp 0.08 --escape_mode window --escape_ratio 2.0 --recovery 2"; mode=g1 ;;
      sota_la_oracle) EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 6 --escape_min_disp 0.10 --escape_mode window --escape_ratio 1.5 --recovery 2 --la_B 32 --la_umax 0.05 --la_sigma 0.03 --la_horizon 2 --la_discrete 1 --rec_hold 2"; mode=g1 ;;
      sota_onm) case $L in h*) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1 --onmanifold_n 8 --project_tau 0.5" ;;
                        *)  EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4 --onmanifold_n 8 --project_tau 0.5" ;; esac; mode=g1 ;;
      sota_onm_np) case $L in h*) EX="$G1 $CBF --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1 --onmanifold_n 8" ;;
                           *)  EX="$G1 $CBF --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4 --onmanifold_n 8" ;; esac; mode=g1 ;;
      *) echo "unknown arm $arm" >&2; continue ;;
    esac
    name=groot_${L}_${arm}_s$seed
    for shard in "libero_spatial 0-3" "libero_spatial 4-6" "libero_spatial 7-9" "libero_object 0-3" "libero_object 4-6" "libero_object 7-9"; do
      suite=${shard%% *}; tasks=${shard#* }
      case "$suite $mode" in
        "libero_spatial g1") port=5561 ;; "libero_spatial none") port=5562 ;;
        "libero_object g1") port=5563 ;;  "libero_object none") port=5564 ;;
      esac
      # --suite must come first: queue_run.sh counts / places jobs by the pattern "python ../eval_obstacle.py --suite"
      echo "$name|$suite|$tasks|--suite $suite --tasks $tasks --episodes ${EPISODES:-10} --policy groot --groot_port $port --replan_steps 8 --baseline_dir ../runs/baseline_groot --calib $CG --out ../runs/$name --torch_seed $seed $LAY $EX"
    done
  done
done
