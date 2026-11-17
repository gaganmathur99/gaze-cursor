"""Focus tracking: time per screen + optional gaze heatmaps."""

import os
import time
from typing import Dict, List

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class FocusTracker:
    HEAT_W, HEAT_H = 96, 54  # heatmap accumulation resolution per screen

    def __init__(self, monitors):
        self.monitors = {m.index: m for m in monitors}
        self.time_by_screen: Dict[int, float] = {m.index: 0.0 for m in monitors}
        self.heat: Dict[int, np.ndarray] = {
            m.index: np.zeros((self.HEAT_H, self.HEAT_W), dtype=np.float64)
            for m in monitors
        }
        self.switches = 0
        self._last_t = time.perf_counter()
        self._last_screen = None
        self.start = self._last_t

    def update(self, screen: int, fx: float, fy: float):
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        if screen is None:
            return
        self.time_by_screen[screen] = self.time_by_screen.get(screen, 0.0) + dt
        if self._last_screen is not None and screen != self._last_screen:
            self.switches += 1
        self._last_screen = screen
        gx = min(int(fx * self.HEAT_W), self.HEAT_W - 1)
        gy = min(int(fy * self.HEAT_H), self.HEAT_H - 1)
        self.heat[screen][gy, gx] += 1.0

    def report(self) -> str:
        total = max(time.perf_counter() - self.start, 1e-6)
        lines = ["", "=== Focus report ==="]
        lines.append(f"Session length : {total:5.1f}s")
        lines.append(f"Screen switches: {self.switches}")
        lines.append("")
        for idx in sorted(self.time_by_screen):
            t = self.time_by_screen[idx]
            pct = 100.0 * t / total
            bar = "█" * int(pct / 2)
            prim = " (primary)" if self.monitors[idx].is_primary else ""
            lines.append(f"Screen {idx}{prim:>10}: {t:5.1f}s {pct:5.1f}% {bar}")
        return "\n".join(lines)

    def save_heatmaps(self, out_dir: str = "reports") -> List[str]:
        if cv2 is None:
            return []
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for idx, acc in self.heat.items():
            if acc.max() <= 0:
                continue
            norm = (acc / acc.max() * 255).astype(np.uint8)
            blur = cv2.GaussianBlur(norm, (0, 0), sigmaX=2.0)
            big = cv2.resize(blur, (960, 540), interpolation=cv2.INTER_CUBIC)
            colored = cv2.applyColorMap(big, cv2.COLORMAP_JET)
            path = os.path.join(out_dir, f"heatmap_screen{idx}.png")
            cv2.imwrite(path, colored)
            paths.append(path)
        return paths
