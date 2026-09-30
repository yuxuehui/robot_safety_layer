# safeguide — a plug-in safety layer for flow-matching VLA policies

```
                 ┌──────────────────────────────────────────────────────────────┐
   Obs. ───────► │  Generative policy (flow matching)  ──►  Safety Layer  ──►  │ ──► safe action chunk
                 │        pi0.5 / GR00T N1.7  (frozen)        core/guide.py    │
                 └──────────────────────────────▲───────────────────────────────┘
                                                │  runtime safety constraints = inference-time tasks
                             StaticObstacleTask (pillars, fixtures)   DynamicObstacleTask (human arm)
                             cylinders  → J_s                          tracked + predicted capsules → J_d
```

A plug-in safety layer for frozen flow-matching / diffusion action policies (pi0.5 today, GR00T N1.x next).
At every Euler step of the policy's own action sampler it estimates the clean action chunk, evaluates a
world-frame collision cost on a sphere model of the robot (plus the object it carries) against static cylinders
and moving capsules (a human arm), and pushes the velocity field downhill in that cost. No gradient through the
policy network, no retraining, ~30-50 ms per policy call on top of the policy itself.

The configuration evaluated as **"inference-time guidance"** in the September 2026 deck is
`GuideConfig.sota()` + `SupervisorConfig.pillar_sota()` (static obstacles) / `SupervisorConfig.hand_sota()` (moving hand):
CBF barrier cost anchored at the measured chunk-start clearance, held-object spheres, Jacobian arm-motion model,
post-gate arm-margin release and deadlock escape, replanning every 5 control steps.

## Layout

```
safeguide/
  api.py                 SafeGuide / SafetyLayer facade (adapter + robot + scene -> installed guided policy)
  tasks.py               the two runtime safety constraints: StaticObstacleTask, DynamicObstacleTask, compose()
  core/
    geometry.py          SDFs (cylinder, capsule; numpy + torch, same formulas)
    cost.py              ActionMap / DeltaEEFMap, ChunkCostContext, clearance and cost functions (hinge / cbf / energy)
    flow.py              FlowSpec: time convention of the flow (OPENPI_FLOW t:1->0, GROOT_FLOW t:0->1)
    guide.py             GuideConfig, Guide: the guided Euler integration + escape + diagnostics
    supervisor.py        SupervisorConfig, Supervisor: per-chunk state machine (release, escape, held object) -> ctx
  adapters/
    base.py              FlowPolicyAdapter interface
    pi05_openpi.py       Pi05Adapter (openpi PyTorch pi0 / pi0.5)          verified bit-identical to the research code
    groot_n1.py          GrootN1Adapter (Isaac-GR00T N1.x action head)     skeleton, untested: verify against N1.7 source
  robot/
    base.py              RobotModel interface
    mujoco_panda.py      Panda in robosuite/MuJoCo: gripper + link5/6/7 spheres, Jacobian motion matrices, held object
  scene/
    base.py              Scene interface, StaticCylinders, CompositeScene
    human_arm.py         HumanArmTrack: tracked (elbow, fingertip) -> two predicted, inflated capsules
```

Benchmark-side glue (LIBERO specific) stays outside the package: `obstacle.PillarScene` (pillars installed in the
env), `hand.HandCapsuleScene` (scripted human arm + its prediction), and the evaluator `eval_obstacle.py`.

## The two inference-time tasks and their cost functions

Shared notation: robot collision spheres p with centres x_hp at chunk step h (gripper 8, forearm/wrist links 15, held
object ≤ 3), radii r_p, margins m_p (1.0 cm gripper / object, 1.5 cm arm), step weights w_h (1 on the executed prefix,
0.3 on the tail), d_safe = 3 cm, γ = 0.3, v_ref = 1 cm. Sphere motion follows the end effector through the Jacobian
model x_hp = x_0p + M_p (p_h − p_0); the end-effector path p_h is the action map applied to the chunk estimate â.

**Task 1 — static obstacles (`StaticObstacleTask`)**: z-aligned cylinders k = (c_k, R_k, H_k).

    c_khp = SDF_cyl_k(x_hp) − r_p                                            clearance
    f_khp = min(d_safe, m_p + (1−γ)^(h+1) · relu(c0_kp − m_p))                barrier floor, c0 = clearance measured now
    J_s(â) = Σ_k Σ_h w_h Σ_p relu(f_khp − c_khp)²

Episode heuristics: arm-margin release once the end effector is past the gate line (+3 cm hysteresis; arm margin →
0), deadlock escape (guidance active on 4 consecutive chunks and end effector moved < 2 cm → lift 1.5 cm per step for
2 chunks).

**Task 2 — dynamic obstacle, a human arm (`DynamicObstacleTask`)**: two capsules (elbow→wrist r = 4 cm,
wrist→fingertip r = 4.5 cm) tracked once per control step (`HumanArmTrack.observe`), poses predicted for the chunk
steps h = 1..H by constant-velocity extrapolation (or static / oracle for analysis), radii inflated by a human-safety
margin plus a reaction-time allowance:

    R_k ← R_k + m_hand + |v_tip| · τ            ("tight": m_hand = 0, τ = 0.1 s;  "safe": 3 cm, 0.2 s)
    c_khp = SDF_cap(x_hp; a_k(h), b_k(h)) − r_p − R_k
    f_khp = same barrier, c0 from the current capsule pose (h = 0)
    J_d(â) = Σ_k Σ_h w_h Σ_p relu(f_khp − c_khp)²

No release, no escape: retreating and waiting for the arm to pass is the desired behaviour (the escape lifted the
gripper 40–50 cm while waiting).

Both costs are applied through the same push at every Euler step of the policy's flow:

    g = ∂J/∂â,   v ← v + λ · min(1, max violation / v_ref) · g / RMS(g)        (λ = 1, no gradient through the policy)

With both tasks active the scene is their union and J = J_s + J_d; `split_costs()` and the `cost_static` /
`cost_dynamic` entries of `Guide.last_log` report the two parts.

```python
import safeguide as sg
layer = sg.compose(adapter, robot, sg.StaticObstacleTask(obstacle.PillarScene(env))).install()    # task 1
layer = sg.compose(adapter, robot, sg.DynamicObstacleTask.tight()).install()                     # task 2
layer = sg.compose(adapter, robot, static_task, dynamic_task).install()                          # both
# per control step (task 2):  dynamic_task.track.observe(elbow, fingertip)
```

## The four interfaces

| Interface | Implement per | Must provide |
|---|---|---|
| `FlowPolicyAdapter` | model family | `prepare(obs) -> cond` (conditioning computed once per chunk), `velocity(cond, x_t, t)` (one velocity-field call), `sample_noise`, `flow` (FlowSpec), `action_horizon`, `action_map()`, `install(guide)` |
| `ActionMap` | action convention | `eef_positions(a_norm, T)` (normalised chunk -> EEF positions), `world_offset_to_action(off, T)` (for the escape). `DeltaEEFMap(lo, hi, G)` covers delta-EEF actions with per-dim affine normalisation (openpi quantiles or GR00T min/max) and a calibrated gain `G`. Joint-space embodiments need a Jacobian-based map. |
| `RobotModel` | robot / simulator | `eef_pos()`, `points(include_arm, held, arm_motion) -> (pts, radii, alphas, groups, motion)`, `held_object(gripper_closed)` |
| `Scene` | obstacle source | `cylinders() -> (centers, radii, half_heights)`, `capsules(H) -> (cap_a, cap_b, cap_r)` with predicted poses at chunk steps 0..H, radii already inflated by the safety margin |

## Usage

```python
import safeguide as sg

adapter = sg.Pi05Adapter(policy, norm_stats, G)              # openpi Policy, its norm stats, calibrated 3x3 gain
guide = sg.SafeGuide(adapter, sg.MujocoPandaRobot(env), scene,
                     sg.GuideConfig.sota(), sg.SupervisorConfig.pillar_sota(exec_steps=5)).install()
for episode in episodes:
    guide.reset_episode(gate_line)                            # (c_xy, d_xy, offset) or None
    while not done:
        if not plan:
            guide.before_chunk(gripper_closed)                # builds the cost context from robot + scene
            chunk = policy.infer(obs)["actions"]              # the policy's own call, now guided
            guide.after_chunk()
            plan.extend(chunk[:5])
        env.step(plan.popleft())
```

`eval_obstacle.py --engine auto` routes `--guidance g1` and `--guidance none` through the package and the research
variants (steer / oc / car / mppi) through the legacy `guidance.GuidedSampler`, which imports its cost primitives from
`safeguide.core.cost` so both share one definition of clearance and cost.

## Equivalence check (2026-09-30, pod3, libero_object task 0, seed 0)

`eqv_compare.py` compares two run dirs episode by episode (result records + executed action / EEF trajectories).

| Comparison | Result |
|---|---|
| gate17 SOTA: `--engine legacy` vs `--engine safeguide` | 2/2 episodes bit-identical (1 skipped placement identical) |
| gate17 SOTA: `--engine legacy` vs archived `obs7_gate17_cbfjac_relesc_s0` (2026-09-28) | bit-identical |
| gate17 unguided: legacy vs safeguide, and vs archived `obs7_gate17_none_s0` | trajectories bit-identical (archived records lack two newer keys) |
| hand sweep SOTA tight margin: legacy vs safeguide, and vs archived `obs7_hsweepvis_sota_cv_t_s0` (2026-09-29) | 3/3 episodes bit-identical |
| hand reach SOTA safe margin, oracle prediction: legacy vs safeguide, and vs archived `obs7_hreachvis_sota_or_s0` | 3/3 episodes bit-identical |
| re-check after the two-task refactor (`HumanArmTrack`, `tasks.py`): gate SOTA, hand cv tight, hand oracle | all bit-identical |

## Running with GR00T N1.7 (bridge)

GR00T N1.7 needs Python 3.12 / torch 2.9 (`Isaac-GR00T/.venv`), the LIBERO evaluator lives in the openpi Python 3.11
venv, so the two talk over zmq (msgpack + msgpack_numpy, the wire format of gr00t's own PolicyServer):

```
evaluator (openpi venv)                                  GR00T venv
eval_obstacle.py --policy groot                          safeguide/server/groot_server.py
  Supervisor next to the sim -> ctx ──obs + ctx──►       Gr00tPolicy with the action head's denoising loop replaced
  RemoteGuidedPolicy.infer()  ◄── chunk (16x7) + log     by Guide (GrootN1Adapter); decode_action un-normalises
```

```bash
# server (GR00T venv), one per checkpoint / guidance mode
cd $WS/0Xuehui/Isaac-GR00T && PYTHONPATH=$PROJ CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
    $PROJ/safeguide/server/groot_server.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_spatial --port 5556 --guidance g1
# obstacle-free baselines (needed for pillar / hand placement and for calibrating G), then the benchmark
.venv/bin/python ../eval_libero.py --policy groot --groot_port 5556 --suite libero_spatial --replan_steps 8 --out ../runs/baseline_groot
.venv/bin/python ../eval_obstacle.py --policy groot --groot_port 5556 --baseline_dir ../runs/baseline_groot --replan_steps 8 ...
```

Facts about the LIBERO checkpoints (`nvidia/GR00T-N1.7-LIBERO/<suite>`, read from the config and the source): action
horizon 40 in the model (padded; 16 carry real actions, `DeltaEEFMap.n_valid`), 8 executed by the reference eval,
4 Euler steps, 1000 time buckets, actions x y z roll pitch yaw gripper (7 of 132 padded dims) normalised to [-1, 1] by
q01/q99, gripper decoded in [0, 1] and converted by `remote.libero_action` exactly as the reference LIBERO env does.
Observations: both 256x256 cameras rotated 180 deg, 8-D state (xyz, axis-angle, 2 gripper joints), the task string.

## Porting checklist for GR00T N1.7

1. Read the installed `gr00t/model/action_head/flow_matching_action_head.py`: confirm `get_action`'s denoising loop
   (time direction, `num_inference_timesteps`, `num_timestep_buckets`, `state_encoder`, `future_tokens`,
   `action_encoder`, `model`, `action_decoder`) and fix the names in `adapters/groot_n1.py` accordingly.
2. Action semantics of the fine-tuned embodiment: delta-EEF (LIBERO fine-tunes) -> `DeltaEEFMap(min, max, G)` with a
   freshly calibrated `G` (`calibrate_action_model.py` on unguided rollouts); joint targets -> new `ActionMap`
   through the arm Jacobian.
3. `SupervisorConfig.exec_steps` = the executed prefix length (8 of 16 for GR00T defaults); margins unchanged.
4. Re-tune `scale` / `cbf_vref` for the shorter flow (4 Euler steps instead of 10): the push is applied 2.5x fewer
   times per chunk.
5. Regression: run `eval_obstacle.py` gate17 / hand benchmarks with the new adapter and compare against the
   pi0.5 numbers in the project README; expect the unguided baseline to differ (different policy), the paired
   improvement to persist.
