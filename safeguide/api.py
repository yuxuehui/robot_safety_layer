"""Facade: wire an adapter, a robot model and a scene into a guided policy.

    guide = SafeGuide(Pi05Adapter(policy, norm_stats, G), MujocoPandaRobot(env), scene,
                      GuideConfig.sota(), SupervisorConfig.pillar_sota(exec_steps=5)).install()
    for episode:
        guide.reset_episode(gate_line)
        while not done:
            if not plan:
                guide.before_chunk(gripper_closed)          # cost context for this chunk
                chunk = policy.infer(obs)["actions"]        # the policy's own call, now guided
                guide.after_chunk()
"""
from .core.guide import Guide, GuideConfig
from .core.supervisor import Supervisor, SupervisorConfig


class SafeGuide:
    def __init__(self, adapter, robot, scene, guide_cfg: GuideConfig | None = None, sup_cfg: SupervisorConfig | None = None,
                 cost_fn=None):
        """cost_fn: optional user-defined constraint cost_fn(a_hat, ctx, T) -> (J (B,), violation (B,) in metres);
        None = the built-in obstacle cost of `scene` (cylinders + capsules)."""
        self.adapter = adapter
        self.guide = Guide(adapter, guide_cfg or GuideConfig.sota(), cost_fn=cost_fn)
        self.sup = Supervisor(robot, scene, adapter.action_map(), sup_cfg or SupervisorConfig(),
                              horizon=adapter.action_horizon)

    def install(self):
        self.adapter.install(self.guide)
        return self

    def uninstall(self):
        self.adapter.uninstall()

    def reset_episode(self, gate_line=None):
        self.guide.reset_episode()
        self.sup.reset(gate_line)

    def before_chunk(self, gripper_closed: bool):
        ctx = self.sup.before_chunk(gripper_closed)
        self.guide.ctx, self.guide.escape_off = ctx, self.sup.escape_off
        return ctx

    def after_chunk(self):
        self.sup.after_chunk(self.guide.last_log)

    @property
    def last_log(self):
        return self.guide.last_log


# Name used in the SHARE@NTU P1.3 diagram: Obs -> generative policy -> Safety Layer (+ runtime safety constraints) -> safe chunk
SafetyLayer = SafeGuide
