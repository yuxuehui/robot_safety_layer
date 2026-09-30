"""Obstacle sources. A Scene answers two questions per chunk: which static vertical cylinders are there, and which
moving capsules, with their predicted poses over the chunk. Sim ground truth today; a perception front end
(point cloud -> primitives, hand tracker -> capsules + constant-velocity prediction) implements the same two calls."""
import numpy as np


class Scene:
    def cylinders(self):
        """(centers (K, 3), radii (K,), half_heights (K,)) of z-aligned cylinders; K may be 0."""
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

    def capsules(self, horizon):
        """None, or (cap_a (K2, H+1, 3), cap_b (K2, H+1, 3), cap_r (K2,)): capsule axis end points at chunk steps
        0 (now) .. H (predicted), radii already inflated by any safety margin."""
        return None


class StaticCylinders(Scene):
    def __init__(self, centers, radii, half_heights):
        self.c, self.r, self.h = np.asarray(centers, float).reshape(-1, 3), np.asarray(radii, float), np.asarray(half_heights, float)

    def cylinders(self):
        return self.c, self.r, self.h


class CompositeScene(Scene):
    """Union of several scenes."""

    def __init__(self, *scenes):
        self.scenes = scenes

    def cylinders(self):
        parts = [s.cylinders() for s in self.scenes]
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))

    def capsules(self, horizon):
        parts = [c for c in (s.capsules(horizon) for s in self.scenes) if c is not None]
        if not parts:
            return None
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))
