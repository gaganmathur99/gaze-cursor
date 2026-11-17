"""Radar overlay: a mini-map of all monitors with a live gaze dot."""

import cv2
import numpy as np


class Radar:
    def __init__(self, monitors, target_w: int = 640, margin: int = 24):
        self.monitors = monitors
        xs0 = min(m.x for m in monitors)
        ys0 = min(m.y for m in monitors)
        xs1 = max(m.x + m.width for m in monitors)
        ys1 = max(m.y + m.height for m in monitors)
        self.min_x, self.min_y = xs0, ys0
        span_x, span_y = xs1 - xs0, ys1 - ys0
        self.scale = (target_w - 2 * margin) / span_x
        self.margin = margin
        self.w = target_w
        self.h = int(span_y * self.scale) + 2 * margin
        self.window = "gaze radar (q to quit)"
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)

    def _to_px(self, x: int, y: int):
        px = int((x - self.min_x) * self.scale) + self.margin
        py = int((y - self.min_y) * self.scale) + self.margin
        return px, py

    def draw(self, cursor_xy=None, active_screen=None, face_found=True):
        canvas = np.full((self.h, self.w, 3), 25, dtype=np.uint8)
        for m in self.monitors:
            x0, y0 = self._to_px(m.x, m.y)
            x1, y1 = self._to_px(m.x + m.width, m.y + m.height)
            active = (active_screen == m.index)
            color = (60, 180, 75) if active else (90, 90, 90)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            label = f"S{m.index}" + ("*" if m.is_primary else "")
            cv2.putText(canvas, label, (x0 + 6, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if cursor_xy is not None:
            cx, cy = self._to_px(*cursor_xy)
            cv2.circle(canvas, (cx, cy), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (cx, cy), 13, (255, 255, 255), 2)
        status = "tracking" if face_found else "no face"
        scolor = (60, 220, 60) if face_found else (60, 60, 220)
        cv2.putText(canvas, status, (self.margin, self.h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, scolor, 2)
        cv2.imshow(self.window, canvas)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self):
        cv2.destroyWindow(self.window)
