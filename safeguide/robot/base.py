"""What the cost needs from the robot: a sphere model of the links that can hit things, where the end effector is,
how each sphere moves when the end effector moves, and which task object is being carried."""


class RobotModel:
    def eef_pos(self):
        """(3,) world position of the end-effector reference point (the point the action map moves)."""
        raise NotImplementedError

    def points(self, include_arm=True, held=None, arm_motion="jac"):
        """Collision spheres for the chunk that starts now.
        Returns (pts (P, 3), radii (P,), alphas (P,), groups [P str], motion (P, 3, 3) or None):
          alphas   heuristic motion weight: sphere moves by alpha * dEEF (used when motion is None)
          groups   'grip' / 'hand' / 'hand_w' (gripper), 'link5' / 'link6' / 'link7' (arm), 'obj' (held object)
          motion   per-sphere linear motion model d p_sphere = M_p d p_ee from the Jacobians (arm_motion == 'jac')
        held: name of the carried task object whose collision spheres ride rigidly with the hand (or None)."""
        raise NotImplementedError

    def held_object(self, gripper_closed: bool):
        """Name of the task object currently carried (None if none / gripper open)."""
        raise NotImplementedError
