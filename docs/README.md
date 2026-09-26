# Beads Out local agent

Small Windows prototype using Pillow and NumPy for classical image processing. It detects the visible board, cube colors, raised/depressed relief, locked pieces, numbered special tiles, the visible right-hand feed, and a conservative next move. No model downloads or game modification are used.

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

The detector has been exercised against the screenshots in `root/img/`. On Level 59 it finds 20 cells, six raised cells, two locked pieces, two `2` tiles, and the `Very Hard` badge. It also identifies the Level 57 ice tiles marked `200`, while leaving the hidden cubes' colors unknown. The RH/DNH score uses the bright neutral rim around each box, not its color.

The current feed reader recognizes visible leading beads on the right conveyor; when that conveyor is hidden it abstains. Its upcoming-color read is a simple sample heuristic and has not been validated on a live phone. Grid recovery assumes mostly regular rows and columns. Difficulty recognition currently covers the centered `Very Hard` badge shown in the supplied sample. Completion/failure screens are not advanced automatically.

Live capture, tap verification, sustained level completion, and throughput remain unverified because no ADB device was connected during this pass. Run the live command once the phone is available; use the saved before/after frames and timing fields to validate the hardware gates.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "root/src"
py -m unittest discover -s root/tests -v
```

The sample images live in `root/img/`, the Python package in `root/src/beads_bot/`, and generated debug artifacts in `root/debug/`.
