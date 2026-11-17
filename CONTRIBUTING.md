# Contributing

Issues and pull requests are welcome. For changes, please describe the behavior
being changed and how you verified it.

## Development setup

1. Install Python 3.10+ and [uv](https://docs.astral.sh/uv/).
2. Clone the repository and enter its directory.
3. Run `uv sync` to create the environment from the lock file.

Run the app with `uv run gaze-cursor --help`. Hardware-dependent flows need a
webcam, camera permission, and (for calibration and cursor control) a display.
Do not commit `calibration.json` or generated files under `reports/`.

Before submitting a pull request, run:

```bash
uv run python -m compileall -q gaze_cursor
uv run gaze-cursor --help
```

## Pull requests

- Keep changes focused and explain user-visible behavior in the pull request.
- Update the README when setup or command-line behavior changes.
- Include or update automated tests when changing testable behavior.
