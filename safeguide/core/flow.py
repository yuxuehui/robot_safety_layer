"""Flow-matching time conventions. The guided integrator only needs: the time grid, the Euler step, the clean
estimate a_hat from (x_t, t, v), how far along the flow is, and the sign that makes a velocity change move x
downhill in the cost."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class FlowSpec:
    name: str
    noise_at_one: bool  # True: t=1 is noise, t=0 is data (openpi pi0/pi0.5); False: t=0 noise, t=1 data (GR00T N1.x)

    def time(self, i, num_steps):
        """Flow time of Euler step i (Python float, same arithmetic as the models' own samplers)."""
        return 1.0 - i / num_steps if self.noise_at_one else i / num_steps

    def dt(self, num_steps):
        return -1.0 / num_steps if self.noise_at_one else 1.0 / num_steps

    def a_hat(self, x, t, v):
        """Predicted clean chunk from the current state and velocity (rectified flow: straight line to the data)."""
        return x - t * v if self.noise_at_one else x + (1.0 - t) * v

    def noise_level(self, t):
        """1 at pure noise, 0 at data."""
        return t if self.noise_at_one else 1.0 - t

    def progress(self, t):
        """Fraction of the way to the data end (used by the 'linear' guidance schedule)."""
        return 1.0 - t if self.noise_at_one else t

    @property
    def descent_sign(self):
        """v <- v + sign * lam * g moves x along -g under x <- x + dt v."""
        return 1.0 if self.noise_at_one else -1.0


# pi0 / pi0.5 (openpi): x_t = t eps + (1-t) a, v = eps - a, Euler dt = -1/N from t = 1.
OPENPI_FLOW = FlowSpec("openpi", noise_at_one=True)
# GR00T N1.x flow-matching action head: x_t = (1-t) eps + t a, v = a - eps, Euler dt = +1/N from t = 0.
GROOT_FLOW = FlowSpec("groot", noise_at_one=False)
