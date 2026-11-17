"""Move the OS cursor across monitors.

On macOS, `pynput` silently fails to move the cursor without Accessibility
permission and is unreliable across displays, so we use Quartz's
CGWarpMouseCursorPosition directly (no special permission required). Other
platforms fall back to pynput.
"""

import sys

from .monitors import Monitor


class CursorController:
    def __init__(self):
        self._warp = None
        if sys.platform == "darwin":
            try:
                import Quartz

                self._warp = Quartz.CGWarpMouseCursorPosition
                self._associate = Quartz.CGAssociateMouseAndMouseCursorPosition
            except Exception:
                self._warp = None
        if self._warp is None:
            from pynput.mouse import Controller

            self._mouse = Controller()

    def move_to(self, x: int, y: int):
        if self._warp is not None:
            self._warp((float(x), float(y)))
            # Re-associate so the hardware mouse keeps control after the warp.
            self._associate(True)
        else:
            self._mouse.position = (x, y)

    def move_to_screen_center(self, monitor: Monitor):
        self.move_to(*monitor.center)
