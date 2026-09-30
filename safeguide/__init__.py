"""safeguide: inference-time obstacle-avoidance guidance for flow-matching VLA policies (pi0.5, GR00T N1.x, ...).

Layers (each replaceable on its own):
  adapters/  FlowPolicyAdapter  - one per model: conditioning, one velocity-field call, flow time convention,
                                  action normalisation, how to hook the policy's sampler
  robot/     RobotModel         - one per robot/simulator: collision spheres, EEF, Jacobian motion model, held object
  scene/     Scene              - obstacle source: static cylinders and/or moving capsules with predictions
  core/      cost, Guide, Supervisor - model-agnostic: world-frame CBF cost, guided flow integration, per-chunk
                                  state machine (arm-margin release, deadlock escape, held object)
  api.py     SafeGuide / SafetyLayer - facade wiring the four together (the 'Safety Layer' box of the P1.3 diagram)
  tasks.py   StaticObstacleTask, DynamicObstacleTask - the two runtime safety constraints (pillars; human arm)

The package is imported lazily so that light scripts (placement, metrics) that only need the geometry do not pull
in torch.
"""
import importlib

_LAZY = {
    "Guide": ".core.guide", "GuideConfig": ".core.guide",
    "Supervisor": ".core.supervisor", "SupervisorConfig": ".core.supervisor", "build_ctx": ".core.supervisor",
    "ChunkCostContext": ".core.cost", "ActionMap": ".core.cost", "DeltaEEFMap": ".core.cost",
    "chunk_eef_positions": ".core.cost", "sphere_positions": ".core.cost", "sphere_clearance": ".core.cost",
    "step_weights": ".core.cost", "obstacle_costs": ".core.cost",
    "FlowSpec": ".core.flow", "OPENPI_FLOW": ".core.flow", "GROOT_FLOW": ".core.flow",
    "FlowPolicyAdapter": ".adapters.base", "Pi05Adapter": ".adapters.pi05_openpi", "GrootN1Adapter": ".adapters.groot_n1",
    "RobotModel": ".robot.base", "MujocoPandaRobot": ".robot.mujoco_panda",
    "Scene": ".scene.base", "StaticCylinders": ".scene.base", "CompositeScene": ".scene.base",
    "SafeGuide": ".api", "SafetyLayer": ".api",
    "HumanArmTrack": ".scene.human_arm",
    "StaticObstacleTask": ".tasks", "DynamicObstacleTask": ".tasks", "RobotMargins": ".tasks", "compose": ".tasks",
    "split_costs": ".tasks", "supervisor_config": ".tasks",
    "RemoteGuidedPolicy": ".remote",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    if name in ("api", "tasks", "remote", "core", "adapters", "robot", "scene", "server"):
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
