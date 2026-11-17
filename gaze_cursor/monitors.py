"""Discover connected monitors and their geometry.

On macOS we read bounds straight from Quartz (CoreGraphics). `screeninfo`
reports flipped/incorrect Y coordinates for secondary displays on macOS, which
sends the cursor to coordinates that don't exist. Using Quartz here keeps the
geometry in the *same* coordinate space that CursorController warps into.
"""

import sys
from dataclasses import dataclass
from typing import List


@dataclass
class Monitor:
    index: int
    x: int          # top-left x in the global display space
    y: int          # top-left y in the global display space
    width: int
    height: int
    is_primary: bool

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def point(self, fx: float, fy: float) -> tuple[int, int]:
        """Map fractional (0..1, 0..1) coords within this monitor to absolute pixels."""
        fx = min(max(fx, 0.0), 1.0)
        fy = min(max(fy, 0.0), 1.0)
        return (int(self.x + fx * self.width), int(self.y + fy * self.height))


def _macos_monitors() -> List[Monitor]:
    import Quartz

    err, ids, _ = Quartz.CGGetActiveDisplayList(16, None, None)
    main = Quartz.CGMainDisplayID()
    monitors = []
    for i, d in enumerate(ids):
        b = Quartz.CGDisplayBounds(d)
        monitors.append(
            Monitor(
                index=i,
                x=int(b.origin.x),
                y=int(b.origin.y),
                width=int(b.size.width),
                height=int(b.size.height),
                is_primary=(d == main),
            )
        )
    return monitors


def _screeninfo_monitors() -> List[Monitor]:
    from screeninfo import get_monitors

    monitors = []
    for i, m in enumerate(get_monitors()):
        monitors.append(
            Monitor(
                index=i,
                x=m.x,
                y=m.y,
                width=m.width,
                height=m.height,
                is_primary=bool(getattr(m, "is_primary", False)),
            )
        )
    return monitors


def list_monitors() -> List[Monitor]:
    if sys.platform == "darwin":
        try:
            monitors = _macos_monitors()
        except Exception:
            monitors = _screeninfo_monitors()
    else:
        monitors = _screeninfo_monitors()
    if not monitors:
        raise RuntimeError("No monitors detected.")
    return monitors


if __name__ == "__main__":
    for m in list_monitors():
        tag = " (primary)" if m.is_primary else ""
        print(f"Screen {m.index}: {m.width}x{m.height} at ({m.x},{m.y}){tag}")
