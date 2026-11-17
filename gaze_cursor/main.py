"""Gaze-driven multi-monitor cursor.

Usage:
    python -m gaze_cursor.main --calibrate      # 9-point grid, look where prompted
    python -m gaze_cursor.main                  # within-screen gaze control + report
    python -m gaze_cursor.main --mode screen    # just jump between screens
    python -m gaze_cursor.main --radar          # show the mini-map overlay
    python -m gaze_cursor.main --heatmap        # save gaze heatmaps on exit
    python -m gaze_cursor.main --list           # print detected monitors
"""

import argparse
import sys
import time
from collections import deque, Counter

import cv2
import numpy as np

from .monitors import list_monitors
from .gaze import GazeEstimator
from .calibration import CalibrationData, GazeModel, GazeRegressor, GRID
from .controller import CursorController
from .tracker import FocusTracker
from .viz import Radar
from .filters import OneEuro2D

SAMPLES_PER_POINT = 25       # calibration frames per grid point
CONFIDENCE_MIN = 0.55        # ignore screen predictions below this
DWELL_FRAMES = 8             # consecutive agreeing frames before switching screen
FEATURE_WINDOW = 7           # frames of gaze features median-filtered before predict
DEADZONE_PX = 14             # don't move the cursor for jitter smaller than this
# 1€ filter params (on fractional 0..1 coords): lower mincutoff = steadier at rest.
EURO = dict(mincutoff=0.5, beta=0.4, dcutoff=1.0)


def open_camera(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open camera {index}.", file=sys.stderr)
        sys.exit(1)
    return cap


def _capture_samples(cap, est, data, screen_idx, fx, fy, n):
    collected = 0
    while collected < n:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        feat = est.extract(rgb)
        if feat is not None:
            data.add(feat.vector, screen_idx, fx, fy)
            collected += 1
            print(f"\r    captured {collected}/{n}", end="")
        time.sleep(0.01)
    print("  ok")


def calibrate(camera_index: int = 0):
    monitors = list_monitors()
    print(f"Detected {len(monitors)} screen(s). 9-point grid per screen.")
    cap = open_camera(camera_index)
    est = GazeEstimator()
    data = CalibrationData()
    try:
        for m in monitors:
            print(f"\n=== Screen {m.index} ({m.width}x{m.height}) ===")
            for (fx, fy, label) in GRID:
                input(f"  Look at the {label} of screen {m.index}, then press Enter...")
                _capture_samples(cap, est, data, m.index, fx, fy, SAMPLES_PER_POINT)
        data.save()
        print(f"\nSaved {len(data.X)} samples across {len(monitors)} screen(s). Done.")
    finally:
        cap.release()
        est.close()


def run(camera_index=0, mode="point", radar=False, heatmap=False, preview=False):
    monitors = list_monitors()
    mon_by_idx = {m.index: m for m in monitors}
    data = CalibrationData.load()
    clf = GazeModel(); clf.train(data)
    reg = None
    if mode == "point":
        if not data.has_grid:
            print("Calibration has no grid data; falling back to screen mode. "
                  "Re-run --calibrate for within-screen control.")
            mode = "screen"
        else:
            reg = GazeRegressor(); reg.train(data)
    print(f"Models trained. Mode: {mode}.")

    cap = open_camera(camera_index)
    est = GazeEstimator()
    cursor = CursorController()
    tracker = FocusTracker(monitors)
    radar_view = Radar(monitors) if radar else None

    recent = deque(maxlen=DWELL_FRAMES)
    feat_buf = deque(maxlen=FEATURE_WINDOW)
    current_screen = None
    euro = OneEuro2D(**EURO)      # filters fractional (fx, fy); reset on screen switch
    last_cursor = None           # last pixel we actually moved to (for deadzone)
    last_t = time.perf_counter()
    print("Running. Look around to move the cursor. Ctrl+C (or 'q' on radar) to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            feat = est.extract(rgb)
            face_found = feat is not None
            cursor_xy = None

            if face_found:
                # Median-filter the feature vector over a short window to kill
                # per-frame outliers before they reach the models.
                feat_buf.append(feat.vector)
                vec = np.median(np.array(feat_buf), axis=0)

                now = time.perf_counter()
                dt = now - last_t
                last_t = now

                screen, conf = clf.predict(vec)
                if conf >= CONFIDENCE_MIN:
                    recent.append(screen)

                # Confirm a screen switch via dwell.
                if len(recent) == recent.maxlen:
                    winner, count = Counter(recent).most_common(1)[0]
                    if count >= recent.maxlen - 1 and winner != current_screen:
                        current_screen = winner
                        euro = OneEuro2D(**EURO)  # reset filter on switch
                        last_cursor = None

                if current_screen is not None:
                    m = mon_by_idx[current_screen]
                    if mode == "point" and reg is not None:
                        fx, fy = reg.predict(vec, current_screen)
                        fx, fy = euro(fx, fy, dt)
                        target = m.point(fx, fy)
                        tracker.update(current_screen, fx, fy)
                    else:
                        target = m.center
                        tracker.update(current_screen, 0.5, 0.5)

                    # Fixation deadzone: ignore sub-threshold jitter.
                    if last_cursor is None or abs(target[0] - last_cursor[0]) + \
                            abs(target[1] - last_cursor[1]) >= DEADZONE_PX:
                        cursor.move_to(*target)
                        last_cursor = target
                    cursor_xy = last_cursor

            if radar_view is not None:
                if radar_view.draw(cursor_xy, current_screen, face_found):
                    break
            if preview:
                cv2.imshow("camera (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        est.close()
        if radar_view is not None:
            radar_view.close()
        if preview:
            cv2.destroyAllWindows()
        print(tracker.report())
        if heatmap:
            paths = tracker.save_heatmaps()
            if paths:
                print("\nHeatmaps saved:")
                for p in paths:
                    print(f"  {p}")


def main():
    p = argparse.ArgumentParser(description="Gaze-driven multi-monitor cursor")
    p.add_argument("--calibrate", action="store_true", help="run 9-point calibration")
    p.add_argument("--list", action="store_true", help="list monitors and exit")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--mode", choices=["point", "screen"], default="point",
                   help="within-screen gaze control (point) or screen jumping (screen)")
    p.add_argument("--radar", action="store_true", help="show mini-map overlay")
    p.add_argument("--heatmap", action="store_true", help="save gaze heatmaps on exit")
    p.add_argument("--preview", action="store_true", help="show webcam preview window")
    args = p.parse_args()

    if args.list:
        for m in list_monitors():
            tag = " (primary)" if m.is_primary else ""
            print(f"Screen {m.index}: {m.width}x{m.height} at ({m.x},{m.y}){tag}")
        return
    if args.calibrate:
        calibrate(args.camera)
        return
    run(args.camera, mode=args.mode, radar=args.radar,
        heatmap=args.heatmap, preview=args.preview)


if __name__ == "__main__":
    main()
