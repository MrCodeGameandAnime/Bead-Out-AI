# Beads Out local agent

Small Windows prototype using Pillow and NumPy for classical image processing. It detects visible board cells, cube colors, raised/depressed relief, independent lock markers, and special tiles. Unknown feed colors and unreadable special values remain unknown while the policy ranks plausible tile actions.

## Install

From PowerShell at the repository root:

```powershell
py -m pip install -r requirements.txt
$env:PYTHONPATH = "root/src"
```

To control the phone, enable USB debugging, connect and unlock it, and make `adb.exe` available on `PATH` or pass its path with `--adb`. `scrcpy` is useful for mirroring and checking the phone, while capture and taps use ADB directly.

## Offline screenshot analysis

```powershell
python -m beads_bot root/img/level59_very_hard_locked_special_mechanics.png --debug-dir root/debug
```

The JSON report contains the full ranked candidate list, each candidate's requirements and assumptions, and a proposed action. The annotated screenshot labels candidate ranks while retaining the RH/DNH border colors. Offline analysis does not send input.

## Live use

Capture and rank candidates without tapping:

```powershell
python -m beads_bot --live --debug-dir root/debug
```

Start continuous play, observing after every action:

```powershell
python -m beads_bot --live --execute --debug-dir root/debug
```

`--execute` keeps choosing and attempting the best untried candidate until the level completes, a confirmed failure occurs, the UI becomes unrecognized, device execution fails, no plausible action remains, or all candidates for a stable board produce no change. Unknown feed, low confidence, unreadable special values, and unfamiliar mechanics lower candidate rank without stopping the run. `--max-moves N` optionally caps gameplay actions; by default the runner is unbounded. `--loop` remains accepted for older command lines; `--execute` itself is now continuous.

When a tap produces no change, the runner tries another candidate for that same board and does not repeat candidates until the board changes. A board change confirms that the action was accepted, not that it was a good strategic move. `ACTION_ACCEPTED` and `NO_CHANGE` are interaction outcomes. `LOCAL_PROGRESS` requires a separate objective-progress signal. `LEVEL_SUCCESS` and `LEVEL_FAILURE` are recorded as stronger run-level outcomes.

After success, the runner advances only when the screen model provides a recognized free continuation control that does not require an ad. It does not tap guessed, paid, or ad controls. The supplied completion screenshot has no recognized safe continuation, so it is recorded as success and the runner stops there.

Use `--serial DEVICE_SERIAL` if multiple phones are connected. Use `--adb C:\path\to\adb.exe` when ADB is not on `PATH`.

## Run records and evidence

Each live invocation writes an incremental journal and captured frames under `root/debug/runs/<run-id>/`. `events.jsonl` keeps before/after states, every candidate and dependency, the selected tile and exact tap, outcome fields, timings, and assumptions. `manifest.json` records the terminal status and links the latest useful steps. Failure and unrecognized-UI runs are marked as failure examples.

`root/debug/mechanic_evidence.jsonl` stores schema-versioned evidence contexts with a mechanic identity and a flexible feature map. New features such as local geometry and neighboring state can be recorded without changing the ledger format. Accepted interaction evidence stays separate from strategic progress; level success/failure is stored with the action sequence.

Feed geometry refinement is deferred until a real progression failure points to feed ordering or outlet geometry. The current detector still reports the feed as unknown, and hidden `?` conveyor regions are not guessed. The agent can still rank and play visible tile candidates.

## Current perception limits

The detector has regressions for the screenshots in `root/img/` and the Level 60 frame at `root/debug/before_001.png`. It recovers cells on the grid lattice when a lock obscures component boundaries, reports lock markers separately from cells, and leaves unreliable special values `UNKNOWN`. The RH/DNH classifier uses tile-local relief cues. Failure-screen perception has no supplied screenshot regression yet; until one is added, an unrecognized screen is retained as `unrecognized_ui` and input stops.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "root/src"
py -m unittest discover -s root/tests -v
```

The sample images live in `root/img/`, the Python package in `root/src/beads_bot/`, and generated run/debug artifacts in `root/debug/`.
