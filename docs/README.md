# Beads Out local agent

Small Windows prototype using Pillow and NumPy for classical image processing. It detects the visible board, cube colors, raised/depressed relief, independent lock markers, and special tiles. Feed order and special-tile values remain unknown until they can be read reliably. No model downloads or game modification are used.

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

The command prints JSON containing the board and one proposed move, and saves an annotated screenshot. It does not send input.

## Live use

Capture and propose a move without tapping:

```powershell
python -m beads_bot --live --debug-dir root/debug
```

Send exactly one tap, capture again, and stop if the board did not change:

```powershell
python -m beads_bot --live --execute --debug-dir root/debug
```

To continue, use `--loop --max-moves 100`. The loop rebuilds the state after every tap and halts on low confidence, a missing legal move, an unchanged board, or a non-game screen. It never taps a lock-marked tile. The tap threshold defaults to 0.65 and can be changed with `--min-confidence`.

Use `--serial DEVICE_SERIAL` if multiple phones are connected. Use `--adb C:\path\to\adb.exe` when ADB is not on `PATH`.

## Current evidence and limits

The detector has been exercised against the screenshots in `root/img/` and the Level 60 regression frame in `root/debug/before_001.png`. It recovers cells on the grid lattice when a lock obscures component boundaries, and reports lock markers separately from cells. Special tiles are identified while their values stay `UNKNOWN` because digit recognition is not reliable enough to report a number. The feed stays unknown until the conveyor outlet and center order are calibrated. The RH/DNH score uses the bright neutral rim around each box, not its color.

Feed recognition remains unresolved, including when beads are visible, and the move policy abstains while the current bead is unknown. Grid recovery assumes mostly regular rows and columns. Difficulty recognition currently covers the centered `Very Hard` badge shown in the supplied samples. Completion/failure screens are not advanced automatically.

Live capture, tap verification, sustained level completion, and throughput remain unverified because no ADB device was connected during this pass. Run the live command once the phone is available; use the saved before/after frames and timing fields to validate the hardware gates.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "root/src"
py -m unittest discover -s root/tests -v
```

The sample images live in `root/img/`, the Python package in `root/src/beads_bot/`, and generated debug artifacts in `root/debug/`.
