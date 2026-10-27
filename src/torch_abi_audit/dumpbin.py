"""Optional ``dumpbin`` backend for PE/COFF symbol names.

``dumpbin`` is the platform-native equivalent of ``nm`` on Windows, but it only
lands on PATH inside a Visual Studio developer prompt, which is not how most
people run the CLI. So it is preferred when present and skipped otherwise,
leaving :mod:`.pe` as the always-available fallback.

Its output is text, so the parsers below are deliberately structural — they key
off the token layout of each block rather than column offsets, which shifts
between image formats (a forwarded export has no RVA column) and dumpbin
versions.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_HEX = re.compile(r"^[0-9A-Fa-f]+$")

_IMPORTS_HEADING = "Section contains the following imports"
_DELAY_HEADING = "Section contains the following delay load imports"
_SUMMARY_HEADING = "Summary"
_FORWARDED = "(forwarded"


class DumpbinError(RuntimeError):
    """``dumpbin`` is missing, failed, or produced output we don't understand."""


@lru_cache(maxsize=1)
def find_dumpbin() -> str | None:
    """Path to ``dumpbin`` on PATH, or ``None`` when it isn't installed there."""
    return shutil.which("dumpbin")


def read_dumpbin(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(imports, exports)`` for ``path`` by shelling out to ``dumpbin``.

    Raises :class:`DumpbinError` when it cannot be run or fails.
    """
    exe = find_dumpbin()
    if exe is None:
        raise DumpbinError("dumpbin not found in PATH")
    return parse_imports(_run(exe, "/imports", path)), parse_exports(
        _run(exe, "/exports", path)
    )


def _run(exe: str, flag: str, path: Path) -> str:
    proc = subprocess.run(
        [exe, "/nologo", flag, str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise DumpbinError(
            f"dumpbin {flag} failed for {path} (exit {proc.returncode})"
        )
    return proc.stdout


def parse_imports(text: str) -> tuple[str, ...]:
    """Named imports from ``dumpbin /imports`` output.

    Regular blocks print ``<hint> <name>``; delay-load blocks prepend the bound
    IAT slot, so ``<address> <hint> <name>``. Ordinal-only imports have no name
    to report and descriptor rows (``... Import Address Table``,
    ``... Characteristics``) are wider or narrower than either shape, so both
    fall out on their own.
    """
    names: list[str] = []
    in_delay = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(_IMPORTS_HEADING):
            in_delay = False
            continue
        if stripped.startswith(_DELAY_HEADING):
            in_delay = True
            continue
        if stripped == _SUMMARY_HEADING:
            break
        parts = line.split()
        if in_delay:
            if len(parts) == 3 and _HEX.match(parts[0]) and _HEX.match(parts[1]):
                names.append(parts[2])
        elif len(parts) == 2 and _HEX.match(parts[0]):
            names.append(parts[1])
    return tuple(names)


def parse_exports(text: str) -> tuple[str, ...]:
    """Exported names from ``dumpbin /exports`` output.

    A plain export is ``<ordinal> <hint> <RVA> <name>``; a forwarded one drops
    the RVA column and gains ``(forwarded to DLL.name)``, so the name is the
    token before that marker.
    """
    names: list[str] = []
    for line in text.splitlines():
        if line.strip() == _SUMMARY_HEADING:
            break
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit() or not _HEX.match(parts[1]):
            continue
        if parts[3].startswith(_FORWARDED):
            names.append(parts[2])
        else:
            names.append(parts[3])
    return tuple(names)
