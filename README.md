# gaze-cursor

Use your laptop webcam to steer the mouse across connected monitors using your
gaze. Calibrate by looking at targets on each screen; the app learns which screen
and (optionally) which location within that screen you are looking at.

Webcam gaze is not pixel-accurate. This project smooths predictions over multiple
frames to make cursor movement more stable.

## Requirements

- macOS is the primary supported platform; other platforms may work but are not
  regularly tested.
- Python 3.10 or later.
- A webcam and at least one display.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment
  and dependency management.

## Setup

```bash
git clone <repository-url>
cd gaze-cursor
uv sync
```

`uv sync` creates the project environment and installs the dependencies from
`uv.lock`. Run project commands with `uv run`.

### macOS permissions

Allow the terminal or app running gaze-cursor to use the camera in **System
Settings → Privacy & Security → Camera**.

On macOS, cursor movement and monitor geometry use Quartz directly, so
Accessibility permission is not required. Other platforms use `screeninfo` and
`pynput`.

## Usage

```bash
# See detected monitors
uv run gaze-cursor --list

# Calibrate: look at each prompted target and press Enter
uv run gaze-cursor --calibrate

# Run within-screen gaze control (requires calibration)
uv run gaze-cursor

# Switch screens only
uv run gaze-cursor --mode screen

# Show a mini-map and live gaze/cursor position
uv run gaze-cursor --radar

# Save per-screen gaze heatmaps to ./reports/ on exit
uv run gaze-cursor --heatmap

# Show a webcam preview window
uv run gaze-cursor --preview
```

Calibration is saved as `calibration.json` in the current working directory.
Heatmaps are written to `reports/`. These are generated, machine-specific files
and are ignored by Git. On exit (Ctrl+C, or `q` in the radar/preview window), a
focus report shows time spent looking at each screen and the number of screen
switches.

## How it works

1. MediaPipe Face Landmarker extracts eye and face landmarks from each webcam
   frame.
2. Calibration records gaze features while you look at a 9-point grid on each
   monitor.
3. scikit-learn models predict the screen and position; smoothing reduces jitter.
4. The cursor is moved to the predicted location using Quartz on macOS or pynput
   on other platforms.

Camera frames are processed locally by the app and are not intentionally recorded
or transmitted.

## Features

- **Within-screen control** (`--mode point`, default): per-screen regressors map
  gaze to a fractional screen position. Requires a 9-point calibration.
- **Screen switching** (`--mode screen`): parks the cursor at the center of the
  selected screen.
- **Radar overlay** (`--radar`): shows the desktop layout and tracking status.
- **Focus tracker**: reports time per screen and can save heatmaps with
  `--heatmap`.

## Tuning

Adjust the constants near the top of [`gaze_cursor/main.py`](gaze_cursor/main.py)
to tune calibration sample counts, prediction confidence, and smoothing. Re-run
`--calibrate` when your seating position, lighting, or monitor layout changes.

## Limitations

- A single webcam cannot reliably infer exact gaze position without per-session
  calibration; the cursor may drift.
- Results are best with even lighting and your face roughly centered in the
  camera.
- One face is tracked at a time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks.

## License

The gaze-cursor source code is licensed under the [MIT License](LICENSE). The
bundled MediaPipe model is a third-party asset; see the
[MediaPipe project](https://github.com/google-ai-edge/mediapipe) for its
licensing and attribution information.
