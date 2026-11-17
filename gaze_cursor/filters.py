"""Signal smoothing for noisy gaze input.

OneEuroFilter: the standard low-latency jitter filter for pointing devices.
It smooths hard when the signal is near-still (kills shake during fixation) and
relaxes the smoothing as movement speeds up (keeps the cursor responsive).
Reference: Casiez et al., "1€ Filter" (CHI 2012).
"""

import math


class OneEuroFilter:
    def __init__(self, freq: float = 30.0, mincutoff: float = 0.6,
                 beta: float = 0.5, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, dt: float | None = None) -> float:
        if dt is not None and dt > 0:
            self.freq = 1.0 / dt
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class OneEuro2D:
    """Two independent 1€ filters for an (x, y) point."""

    def __init__(self, **kwargs):
        self.fx = OneEuroFilter(**kwargs)
        self.fy = OneEuroFilter(**kwargs)

    def __call__(self, x: float, y: float, dt: float | None = None):
        return self.fx(x, dt), self.fy(y, dt)
