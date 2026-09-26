import time
from pathlib import Path

from .capture import _adb_command, resolve_adb


def tap(
    x: int,
    y: int,
    adb_path: str | Path | None = None,
    serial: str | None = None,
    timeout: float = 5.0,
) -> float:
    """Send one Android tap and return the injection latency in milliseconds."""
    adb = resolve_adb(adb_path)
    started = time.perf_counter()
    _adb_command(adb, ["shell", "input", "tap", str(int(x)), str(int(y))], serial, timeout)
    return round((time.perf_counter() - started) * 1000, 2)
