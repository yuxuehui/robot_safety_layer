"""Template: the control loop with the safety layer installed (simulator or real robot).

Per chunk: (1) layer.before_chunk() builds the constraint context from the robot and the scene as they are NOW,
(2) the policy's own inference call runs guided, (3) layer.after_chunk() updates the supervisor, (4) the first
exec_steps actions are executed, then replan. Latency: the layer adds ~30-50 ms per policy call on pi0.5 (10 Euler
steps) and ~10 ms on GR00T N1.7 (4 steps); budget it in the control period.
"""
from collections import deque

import safeguide as sg


def run_episode(policy, adapter, robot, scene, step_env, get_obs, gripper_closed, exec_steps=5, hand_track=None,
                gate_line=None, max_steps=400):
    layer = sg.SafeGuide(adapter, robot, scene, sg.GuideConfig.sota(),
                         sg.SupervisorConfig.pillar_sota(exec_steps=exec_steps)).install()
    # For a moving-human constraint only: sg.SupervisorConfig.hand_sota(exec_steps=exec_steps) (no release / escape).
    try:
        layer.reset_episode(gate_line)  # gate_line = (c_xy, d_xy, offset) for the arm-margin release, or None
        plan, obs, t = deque(), get_obs(), 0
        while t < max_steps:
            if hand_track is not None:
                hand_track.observe(*get_hand_points())  # elbow, fingertip in the robot frame, every control step
            if not plan:
                layer.before_chunk(gripper_closed())
                chunk = policy.infer(obs)["actions"]  # guided by the installed layer
                layer.after_chunk()
                log = layer.last_log  # cost_final, pred_min_clearance, guided_first, sample_ms ...
                if log.get("cost_final", 0.0) > 0.0:
                    pass  # residual predicted violation after guidance: slow down / hold if your safety policy requires
                plan.extend(chunk[:exec_steps])
            obs, done = step_env(plan.popleft())
            t += 1
            if done:
                break
    finally:
        layer.uninstall()


def get_hand_points():
    raise NotImplementedError("return (elbow_xyz, fingertip_xyz) from your hand tracker")
