"""Template: describe a new robot (simulator or real hardware) to the safety layer.

The cost only sees spheres: where they are now, how they move when the end effector moves, and which task object is
carried. Reference implementation: safeguide/robot/mujoco_panda.py (Franka Panda in robosuite / MuJoCo: 8 gripper
spheres, 15 forearm / wrist spheres, up to 3 spheres for the held object, Jacobian motion matrices).
"""
import numpy as np

from safeguide.robot.base import RobotModel


class MyRobot(RobotModel):
    def __init__(self, driver):
        self.driver = driver  # your simulator handle or the real-robot state interface (joint state, EEF pose, FK)

    def eef_pos(self):
        # (1) World position of the end-effector reference point: the SAME point the action map moves, in the SAME
        #     frame as the obstacles reported by the Scene.
        return np.asarray(self.driver.eef_position(), float)

    def points(self, include_arm=True, held=None, arm_motion="jac"):
        # (2) Collision spheres at the start of the chunk.
        q = self.driver.joint_positions()
        pts, radii, groups = [], [], []
        for centre, r in self.driver.gripper_spheres(q):  # fingers, hand, wrist camera ...
            pts.append(centre); radii.append(r); groups.append("grip")
        if include_arm:
            for link, (centre, r) in self.driver.arm_spheres(q):  # the links that can reach the obstacles
                pts.append(centre); radii.append(r); groups.append(link)  # e.g. "link5", "link6", "link7"
        if held is not None:
            for centre, r in self.driver.object_spheres(held):  # the carried object rides rigidly with the hand
                pts.append(centre); radii.append(r); groups.append("obj")
        pts, radii = np.asarray(pts, float), np.asarray(radii, float)
        # (3) Motion model: how each sphere moves for a small end-effector displacement d p_ee.
        #     "jac": M_p = J_p(q) J_ee(q)^+ (3x3 per sphere, from the position Jacobians of the sphere's link and of
        #     the end effector) -> d p_sphere = M_p d p_ee. Gripper / object spheres move rigidly (M = I).
        #     "alpha": cheap fallback, d p_sphere = alpha_p d p_ee with alpha = 1 (gripper, object), ~0.5 (forearm).
        alphas = np.array([1.0 if g in ("grip", "obj") else 0.5 for g in groups])
        motion = None
        if arm_motion == "jac":
            J_ee_pinv = np.linalg.pinv(self.driver.jacobian_position("eef", q))
            motion = np.stack([np.eye(3) if g in ("grip", "obj") else self.driver.jacobian_position(g, q) @ J_ee_pinv
                               for g in groups])
        return pts, radii, alphas, groups, motion

    def held_object(self, gripper_closed):
        # (4) Which task object is carried right now (None if the gripper is open or empty). On a real robot this can
        #     come from the gripper width / force sensor plus the last grasp target.
        return self.driver.grasped_object_name() if gripper_closed else None
