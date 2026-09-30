"""Two inference-time safety tasks on top of the same guided sampler.

Notation: robot collision spheres p (centres x_hp at chunk step h, radii r_p, margins m_p: 1.0 cm gripper / held
object, 1.5 cm arm), chunk steps h = 1..H of which the first `exec_steps` are executed (weights w_h = 1, tail 0.3),
d_safe = 3 cm, gamma = 0.3, v_ref = 1 cm. Sphere motion follows the end effector through the Jacobian model
x_hp = x_0p + M_p (p_h^ee - p_0^ee); the end-effector path p_h^ee is the action map applied to the chunk estimate a_hat.

  Task 1  StaticObstacleTask  - pillars, fixtures: z-aligned cylinders k = (c_k, R_k, H_k)
      clearance   c_khp   = SDF_cyl_k(x_hp) - r_p
      barrier     f_khp   = min(d_safe, m_p + (1 - gamma)^(h+1) relu(c0_kp - m_p)),   c0 = clearance measured now
      cost        J_s(a)  = sum_k sum_h w_h sum_p relu(f_khp - c_khp)^2
      episode     arm-margin release once the EEF is past the gate line (+3 cm hysteresis), deadlock escape
                  (guidance active on 4 chunks and EEF moved < 2 cm -> lift 1.5 cm/step for 2 chunks)

  Task 2  DynamicObstacleTask - a human arm: two capsules (elbow -> wrist, wrist -> fingertip) tracked every step
      prediction  capsule poses (a_k(h), b_k(h)) for h = 1..H by constant-velocity extrapolation of the tracked
                  elbow / fingertip (or static / oracle for analysis)
      inflation   R_k <- R_k + m_hand + |v_tip| tau        (human-safety margin + reaction-time allowance;
                  "tight": m_hand = 0, tau = 0.1 s; "safe": 3 cm, 0.2 s)
      clearance   c_khp   = SDF_cap(x_hp; a_k(h), b_k(h)) - r_p - R_k
      barrier     same discrete-time barrier, c0 from the CURRENT capsule pose (h = 0)
      cost        J_d(a)  = sum_k sum_h w_h sum_p relu(f_khp - c_khp)^2
      episode     no release, no escape: retreating and waiting for the arm to pass is the desired behaviour

Both costs are minimised by the same push  v <- v + lam * min(1, max violation / v_ref) * g / RMS(g),  g = dJ/da_hat,
at every Euler step of the policy's flow (core/guide.py). When both tasks are active the scene is their union and
J = J_s + J_d; `split_costs` reports the two parts separately.
"""
import dataclasses

import torch

from .api import SafeGuide
from .core.cost import chunk_eef_positions, costs_from_clearance, sphere_clearance, sphere_positions
from .core.guide import GuideConfig
from .core.supervisor import SupervisorConfig
from .scene.base import CompositeScene, Scene
from .scene.human_arm import HumanArmTrack


def split_costs(a_norm, ctx, T):
    """{'static': J_s, 'dynamic': J_d} (scalars, summed over the batch) of a normalised chunk a_norm."""
    p = chunk_eef_positions(a_norm, ctx, T)
    clr = sphere_clearance(sphere_positions(p, T), T)  # (K, B, H, P): cylinders first, then capsules
    costs, _ = costs_from_clearance(clr, ctx, T)  # (K, B)
    n_cyl = int(T["obs_centers"].shape[0])
    z = torch.zeros((), device=costs.device)
    return {"static": costs[:n_cyl].sum() if n_cyl else z, "dynamic": costs[n_cyl:].sum() if costs.shape[0] > n_cyl else z}


@dataclasses.dataclass
class RobotMargins:
    """Robot-side parameters shared by both tasks."""
    margin_grip: float = 0.010
    margin_arm: float = 0.015
    margin_obj: float | None = None  # None -> margin_grip
    d_safe: float = 0.03
    cbf_gamma: float = 0.3
    cbf_vref: float = 0.01
    include_arm: bool = True
    arm_motion: str = "jac"
    model_object: bool = True


@dataclasses.dataclass
class StaticObstacleTask:
    """Task 1: static obstacles (cylinders). `scene` yields them (e.g. obstacle.PillarScene(env) or StaticCylinders)."""
    scene: Scene
    arm_release: bool = True
    arm_margin_post: float = 0.0
    release_hyst: float = 0.03
    escape_chunks: int = 4
    escape_min_disp: float = 0.02
    escape_lift: float = 0.015
    escape_len: int = 2

    @staticmethod
    def cost(a_norm, ctx, T):
        return split_costs(a_norm, ctx, T)["static"]


@dataclasses.dataclass
class DynamicObstacleTask:
    """Task 2: a moving human arm. `track` is fed one (elbow, fingertip) observation per control step and predicts
    the inflated capsules over the chunk."""
    track: HumanArmTrack

    @classmethod
    def tight(cls, **kw):
        """Tight human margin (0 cm + 0.1 s): the configuration that won the hand benchmark."""
        return cls(HumanArmTrack(predict="cv", margin=0.0, tau=0.1, **kw))

    @classmethod
    def safe(cls, **kw):
        return cls(HumanArmTrack(predict="cv", margin=0.03, tau=0.2, **kw))

    @property
    def scene(self):
        return self.track

    @staticmethod
    def cost(a_norm, ctx, T):
        return split_costs(a_norm, ctx, T)["dynamic"]


def supervisor_config(tasks, exec_steps=5, margins: RobotMargins | None = None):
    """SupervisorConfig for a set of tasks: robot margins shared, release / escape only if a static task asks for them."""
    m = margins or RobotMargins()
    st = next((t for t in tasks if isinstance(t, StaticObstacleTask)), None)
    return SupervisorConfig(
        exec_steps=exec_steps, d_safe=m.d_safe, cost_type="cbf", margin_grip=m.margin_grip, margin_arm=m.margin_arm,
        margin_obj=m.margin_obj, cbf_gamma=m.cbf_gamma, cbf_vref=m.cbf_vref, include_arm=m.include_arm,
        arm_motion=m.arm_motion, model_object=m.model_object,
        arm_release=bool(st and st.arm_release), arm_margin_post=st.arm_margin_post if st else 0.0,
        release_hyst=st.release_hyst if st else 0.03, escape_chunks=st.escape_chunks if st else 0,
        escape_min_disp=st.escape_min_disp if st else 0.02, escape_lift=st.escape_lift if st else 0.015,
        escape_len=st.escape_len if st else 2)


def compose(adapter, robot, *tasks, exec_steps=5, margins: RobotMargins | None = None, guide_cfg: GuideConfig | None = None):
    """Guided policy for one or both tasks:  compose(adapter, robot, StaticObstacleTask(scene)).install()
                                            compose(adapter, robot, DynamicObstacleTask.tight()).install()
                                            compose(adapter, robot, static_task, dynamic_task).install()"""
    scenes = [t.scene for t in tasks]
    scene = scenes[0] if len(scenes) == 1 else CompositeScene(*scenes)
    return SafeGuide(adapter, robot, scene, guide_cfg or GuideConfig.sota(), supervisor_config(tasks, exec_steps, margins))
