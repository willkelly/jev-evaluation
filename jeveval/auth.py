"""API key acquisition.

The key lives in `pass`, and decrypting it requires a physical YubiKey touch with
a short timeout. That makes the key expensive to fetch -- not in money but in
interrupting the operator -- so this module fetches it as rarely as possible.

Resolution order:

1. ``TYPESAFE_API_KEY`` in the environment.
2. A session cache in ``$XDG_RUNTIME_DIR`` (tmpfs -- RAM-backed, mode 0700,
   discarded at logout).
3. ``pass typesafe.ai/perf``, which prompts for the touch, and whose result is
   then written to the session cache.

The cache is deliberately confined to tmpfs. The operator's instruction was that
holding the key for the session is fine but it must not be written to disk, and
every Bash invocation here starts a fresh shell that cannot inherit an exported
variable from the last one, so a RAM-backed file is what "an environment
variable for the session" actually has to mean.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PASS_ENTRY = "typesafe.ai/perf"
_CACHED: str | None = None


def _session_cache_path() -> Path | None:
    """Path to the tmpfs-backed session cache, or None if there is no tmpfs."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        return None
    d = Path(runtime)
    if not d.is_dir():
        return None
    return d / "jev-eval-api-key"


def _read_cache() -> str | None:
    p = _session_cache_path()
    if p is None or not p.exists():
        return None
    try:
        # Refuse a cache file that is group- or world-readable rather than
        # silently trusting it.
        if p.stat().st_mode & 0o077:
            return None
        value = p.read_text().strip()
    except OSError:
        return None
    return value or None


def _write_cache(key: str) -> None:
    p = _session_cache_path()
    if p is None:
        return
    try:
        # Create with restrictive permissions from the outset; writing first and
        # chmod-ing after would leave a window where the file is readable.
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key)
    except OSError:
        pass


def get_api_key(*, allow_prompt: bool = True) -> str:
    """Return the API key, fetching it at most once per process.

    Raises RuntimeError rather than returning an empty string, so a
    misconfigured run fails at startup instead of producing a few thousand
    logged 403s that look like data.
    """
    global _CACHED
    if _CACHED:
        return _CACHED

    env = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if env:
        _CACHED = env
        return _CACHED

    cached = _read_cache()
    if cached:
        _CACHED = cached
        return _CACHED

    if not allow_prompt:
        raise RuntimeError(
            "No API key available and prompting is disabled. "
            "Run `scripts/prime-key.sh` first to unlock the session cache."
        )

    print(
        f"Fetching API key via `pass {PASS_ENTRY}` -- touch your YubiKey when it blinks.",
        file=sys.stderr,
        flush=True,
    )
    try:
        proc = subprocess.run(
            ["pass", PASS_ENTRY],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`pass` is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`pass` timed out waiting for the YubiKey touch") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"`pass {PASS_ENTRY}` failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[-300:]}"
        )

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"`pass {PASS_ENTRY}` returned nothing")

    _CACHED = lines[0]
    _write_cache(_CACHED)
    return _CACHED


def redact(text: str) -> str:
    """Replace the key with a placeholder anywhere it appears.

    Applied to everything that gets logged or printed, so a key cannot reach the
    JSONL log or the terminal through an echoed request or an error message.
    """
    if not _CACHED:
        return text
    return text.replace(_CACHED, "<TYPESAFE_API_KEY>")
