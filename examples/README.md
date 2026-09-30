# Porting templates

Copy the file that matches what you are porting, fill in the marked pieces, keep the rest of the layer unchanged.

| File | Port target | What you implement |
|---|---|---|
| `adapter_template.py` | a new flow-matching / diffusion policy | `FlowPolicyAdapter`: time convention, one velocity call, noise, conditioning, action map, how the policy's sampler is replaced |
| `robot_template.py` | a new simulator or a real robot | `RobotModel`: end-effector position, collision spheres from FK, per-sphere motion model, held object |
| `scene_template.py` | a new obstacle source (sim ground truth or perception) | `Scene`: static cylinders and/or moving capsules with predicted poses |
| `custom_cost.py` | a new runtime constraint | `cost_fn(a_hat, ctx, T) -> (J, violation)` |
| `control_loop.py` | wiring everything into a control loop (sim or real) | the per-chunk calls: `before_chunk` → policy call → `after_chunk` |

The four interfaces are defined in `safeguide/adapters/base.py`, `safeguide/robot/base.py`, `safeguide/scene/base.py`
and `safeguide/core/cost.py` (`ActionMap`); the reference implementations are `Pi05Adapter`, `GrootN1Adapter`,
`MujocoPandaRobot`, `StaticCylinders` / `HumanArmTrack` and `DeltaEEFMap`.
