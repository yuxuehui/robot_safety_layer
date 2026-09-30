# robot_safety_layer

Inference-time safety layer for frozen flow-matching VLA policies (pi0.5, GR00T N1.7): obstacle avoidance by
gradient guidance on the denoising velocity, evaluated on LIBERO with pillars and a simulated human hand.

```
observation --> frozen flow policy --> safety layer (this repo) --> safe action chunk
                                        ^ runtime safety constraints (static pillars, moving hand)
```

## Method in one line

At every Euler step of the policy's flow, the predicted clean chunk `a_hat` is scored by a barrier cost `J`
(control-barrier-function floor on the clearance of gripper / arm / held-object spheres to the obstacles) and the
velocity is pushed along the normalised gradient `-dJ/da_hat`, so the executed chunk stays on the policy's own
manifold while satisfying the runtime constraint. A supervisor handles arm-margin release, deadlock escape and
per-chunk replanning. Pseudocode, the two constraint definitions (static pillars, moving human hand) and the
adapter interface are in [`safeguide/README.md`](safeguide/README.md).

## Layout

| Path | What it is |
|---|---|
| `safeguide/` | The packaged method. `core/`: flow conventions, cost / barrier, guided Euler sampler, supervisor; `adapters/`: pi0.5 (openpi) and GR00T N1.7; `robot/`: MuJoCo Panda sphere model; `scene/`: pillars, human-arm track; `tasks.py`: static / dynamic obstacle tasks; `api.py`: `SafetyLayer` facade; `remote.py` + `server/`: zmq bridge to a GR00T policy server. |
| `eval_obstacle.py` | LIBERO benchmark loop with pillar / hand layouts, `--policy pi05|groot`, `--engine safeguide|legacy`, demo recording (`--record`). |
| `eval_libero.py` | Obstacle-free baselines (the pillars are placed relative to these paths). |
| `guidance.py`, `obstacle.py`, `hand.py` | Legacy research sampler and scene helpers. `guidance.py` also holds the experimental variants (MPPI, steering, the learned CAR correction vector); the package reproduces the G1 path bit-for-bit. |
| `calibrate_action_model.py`, `calibrate_arm_model.py`, `placement_stats.py` | Action-to-end-effector gain calibration, arm sphere model calibration, pillar placement statistics. |
| `obs7_jobs.sh`, `groot_jobs.sh`, `queue_run.sh`, `groot_servers.sh`, `groot_baseline.sh`, `groot_patch_ckpt.sh` | Benchmark job generation and queueing on the lab pod (paths are pod-specific). |
| `obs7_analysis.py`, `stall_diagnostics.py`, `eqv_compare.py` | Paired analysis (McNemar, task-clustered permutation), stall diagnosis, bit-equivalence check between two run directories. |
| `record_pi05_3way.sh`, `record_groot_demo.sh`, `compose_video.py`, `compose_grid.py`, `make_*_plans.py`, `plan_*.json`, `extract_video_qa.py` | Demo recordings and side-by-side videos (N panels, top-view overlay of the guidance push). |

## Quick start

pi0.5 (openpi, PyTorch) with two pillars forming a gate, the configuration used in the benchmark:

```bash
python eval_obstacle.py --suite libero_spatial --layout gate --gate_half_gap 0.17 --clean_gate 0.075 \
    --guidance g1 --scale 1.0 --grad_through_model 0 --cost_type cbf --margin_grip 0.010 --margin_arm 0.015 \
    --model_object 1 --arm_motion jac --arm_release 1 --escape_chunks 4 \
    --calib runs/baseline/libero_spatial/action_model_calibration.npz --out runs/demo
```

Moving human hand (dynamic obstacle, constant-velocity prediction):

```bash
python eval_obstacle.py --suite libero_spatial --extra_steps 100 --layout hand --hand_motion sweep --hand_visible 1 \
    --guidance g1 --scale 1.0 --grad_through_model 0 --cost_type cbf --margin_grip 0.010 --margin_arm 0.015 \
    --model_object 1 --arm_motion jac --hand_predict cv --margin_hand 0.0 --hand_margin_tau 0.1 \
    --calib runs/baseline/libero_spatial/action_model_calibration.npz --out runs/demo_hand
```

GR00T N1.7: start the policy servers (`bash groot_servers.sh start`, one checkpoint x guidance mode per GPU) and add
`--policy groot --groot_port <port> --replan_steps 8 --baseline_dir runs/baseline_groot` to the same command.

Frozen-policy reference: replace the guidance flags by `--guidance none`. The obstacle-free baselines that the
placements are derived from come from `eval_libero.py`.

The learned CAR correction vector is kept in `guidance.py` (`--guidance car --car_param vector`, legacy engine) for
reference; it is not part of the recommended configuration and none of the commands above enable it.
