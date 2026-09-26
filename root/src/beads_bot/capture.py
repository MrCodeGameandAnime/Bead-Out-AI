import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CapturedFrame:
    image: Image.Image
    elapsed_ms: float


def resolve_adb(adb_path: str | Path | None = None) -> str:
    if adb_path:
        resolved = Path(adb_path)
        if resolved.is_file():
            return str(resolved)
        raise FileNotFoundError(f"ADB executable not found: {resolved}")

    candidates: list[Path] = []
    configured = os.environ.get("ADB_PATH")
    if configured:
        candidates.append(Path(configured))
    scrcpy_dir = os.environ.get("SCRCPY_DIR")
    if scrcpy_dir:
        candidates.append(Path(scrcpy_dir) / "adb.exe")
    for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME", "LOCALAPPDATA"):
        value = os.environ.get(variable)
        if value:
            base = Path(value)
            candidates.append(base / "platform-tools" / "adb.exe")
            candidates.append(base / "Android" / "Sdk" / "platform-tools" / "adb.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("adb") or shutil.which("adb.exe")
    if found:
        return found
    raise FileNotFoundError("ADB was not found; set ADB_PATH or pass --adb.")


def _adb_command(adb: str, arguments: list[str], serial: str | None, timeout: float) -> bytes:
    command = [adb]
    if serial:
        command.extend(["-s", serial])
    command.extend(arguments)
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ADB command failed ({result.returncode}): {detail or command}")
    return result.stdout


def capture_frame(
    adb_path: str | Path | None = None,
    serial: str | None = None,
    timeout: float = 8.0,
) -> CapturedFrame:
    adb = resolve_adb(adb_path)
    started = time.perf_counter()
    payload = _adb_command(adb, ["exec-out", "screencap", "-p"], serial, timeout)
    image = Image.open(BytesIO(payload)).convert("RGB")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return CapturedFrame(image, round(elapsed_ms, 2))
