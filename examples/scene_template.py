"""Template: obstacle source for the layer (simulator ground truth or a perception front end).

A Scene answers two questions per chunk: which static vertical cylinders are there, and which moving capsules with
their predicted poses over the chunk. Reference implementations: safeguide/scene/base.py (StaticCylinders,
CompositeScene) and safeguide/scene/human_arm.py (HumanArmTrack: two capsules + constant-velocity prediction).
"""
import numpy as np

from safeguide.scene.base import CompositeScene, Scene, StaticCylinders
from safeguide.scene.human_arm import HumanArmTrack


class PerceivedPillars(Scene):
    """Static obstacles from perception: fit vertical cylinders to the obstacle point cloud once (or whenever the scene
    changes) and report them; radii are the fitted radii (the safety margins are added by the cost, not here)."""

    def __init__(self):
        self.centers, self.radii, self.half_heights = np.zeros((0, 3)), np.zeros(0), np.zeros(0)

    def update(self, centers, radii, half_heights):
        self.centers = np.asarray(centers, float).reshape(-1, 3)
        self.radii, self.half_heights = np.asarray(radii, float), np.asarray(half_heights, float)

    def cylinders(self):
        return self.centers, self.radii, self.half_heights


def build_scene(hand_tracker=True):
    """Static pillars + a tracked human arm. Feed `track.observe(elbow_xyz, fingertip_xyz)` once per control step
    from your hand / body tracker; the capsules are inflated by margin + |v_tip| * tau inside HumanArmTrack."""
    pillars = PerceivedPillars()
    if not hand_tracker:
        return pillars, None
    track = HumanArmTrack(predict="cv", margin=0.0, tau=0.1, dt=0.05)  # dt = control period in seconds
    return CompositeScene(pillars, track), track


# Simulator ground truth is the same interface: StaticCylinders(centers, radii, half_heights) for known pillars.
