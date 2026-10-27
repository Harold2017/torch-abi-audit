"""Build fixture extensions with whatever native toolchain the host has.

The compiled-fixture tests used to assume a Unix ``cc``/``g++`` command line.
Windows has no such thing: extension modules are built by MSVC, whose
environment (``INCLUDE``, ``LIB``, ``PATH``) only exists inside a developer
prompt. This module finds a compiler — MSVC via ``vswhere`` plus ``vcvars``,
otherwise a GNU-style driver — and hides both command lines behind one call, so
fixtures are built the same way on every CI platform.

Callers get :class:`ToolchainUnavailable` when no compiler exists and should
skip; a compiler that is present but fails is a real failure and raises.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_VC_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"


class ToolchainUnavailable(RuntimeError):
    """No usable C or C++ compiler on this host."""


@dataclass(frozen=True, slots=True)
class Toolchain:
    """A compiler family plus the environment needed to run it."""

    kind: str  # "msvc" | "gnu"
    cc: str | None
    cxx: str | None
    #: Captured ``vcvars`` environment; ``None`` means inherit the process one.
    env: dict[str, str] | None = None

    def driver(self, *, cpp: bool) -> str | None:
        return self.cxx if cpp else self.cc


def extension_suffix() -> str:
    """Filename suffix for a compiled extension module on this host.

    Deliberately not :data:`sysconfig.EXT_SUFFIX`: fixtures are looked up by
    fixed names, and the classifier under test must not depend on its own
    platform-tag parsing.
    """
    return ".pyd" if sys.platform == "win32" else ".so"


def library_suffix() -> str:
    """Filename suffix for a bundled (non-extension) shared library here."""
    if sys.platform == "win32":
        return ".dll"
    return ".dylib" if sys.platform == "darwin" else ".so"


@functools.lru_cache(maxsize=1)
def find_toolchain() -> Toolchain | None:
    """The compiler to build fixtures with, or ``None`` when there is none."""
    if sys.platform == "win32":
        msvc = _find_msvc()
        if msvc is not None:
            return msvc
    for cxx, cc in (("c++", "cc"), ("g++", "gcc"), ("clang++", "clang")):
        cxx_path = shutil.which(cxx)
        cc_path = shutil.which(cc)
        if cxx_path or cc_path:
            return Toolchain(kind="gnu", cc=cc_path, cxx=cxx_path)
    return None


@functools.lru_cache(maxsize=1)
def _msvc_environment() -> dict[str, str] | None:
    """Environment produced by ``vcvars``, or ``None`` if VS isn't installed.

    ``vswhere`` is the only supported way to locate a VS installation, and a
    batch file is the only way to obtain ``cl``'s environment.
    """
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    located = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            _VC_COMPONENT,
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    roots = located.stdout.strip().splitlines()
    if located.returncode != 0 or not roots:
        return None
    vcvars = Path(roots[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        return None
    return _vcvars_environment(vcvars)


def _vcvars_environment(vcvars: Path) -> dict[str, str] | None:
    """Environment after ``vcvars64.bat`` ran, or ``None`` when it fails."""
    cmd = os.environ.get("COMSPEC", "cmd.exe")
    # Passing the batch path as its own argv entry is what makes cmd.exe accept
    # it: a single "call \"C:\...\vcvars64.bat\" && set" string gets mangled by
    # Windows argument quoting and cmd reports the path as not recognised.
    captured = subprocess.run(
        [cmd, "/c", "call", str(vcvars), ">nul", "&&", "set"],
        capture_output=True,
        # `set` is a cmd.exe builtin, so it writes to the pipe in the OEM code
        # page rather than the ANSI one. Decoding as ANSI would turn a profile
        # such as C:\Users\José into U+FFFD and break INCLUDE/LIB later on.
        encoding="oem",
        check=False,
        errors="replace",
    )
    if captured.returncode != 0:
        return None
    return _parse_env_block(captured.stdout)


def _parse_env_block(text: str) -> dict[str, str] | None:
    """Parse ``KEY=VALUE`` lines as printed by ``set``."""
    env: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        # cmd also emits pseudo-variables like "=C:=C:\path" when started there.
        if separator and key and " " not in key:
            env[key] = value
    return env or None


def _find_msvc() -> Toolchain | None:
    env = _msvc_environment()
    if env is None:
        return None
    cl = shutil.which("cl", path=env.get("PATH", ""))
    if cl is None:
        return None
    return Toolchain(kind="msvc", cc=cl, cxx=cl, env=env)


def _python_import_library(*, limited_api: bool = False) -> Path:
    """The interpreter's import library, for linking fixtures against CPython.

    CPython installs two: ``python3XX.lib`` with every symbol, and
    ``python3.lib`` restricted to the stable ABI. Limited-API builds must use
    the latter — the versioned library carries a default-library directive that
    makes the linker demand ``python3.lib`` anyway.
    """
    base = Path(sysconfig.get_config_var("installed_base") or sys.base_prefix)
    libs = base / "libs"
    versioned = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    candidates = ("python3.lib", versioned) if limited_api else (versioned, "python3.lib")
    for candidate in candidates:
        if (libs / candidate).is_file():
            return libs / candidate
    raise ToolchainUnavailable(f"no Python import library under {libs}")


def build_extension(
    *,
    toolchain: Toolchain,
    source: Path,
    output: Path,
    cpp: bool = False,
    includes: Sequence[str] = (),
    defines: Sequence[str] = (),
    stdlib: str | None = None,
    libraries: Sequence[str] = (),
    library_dirs: Sequence[str] = (),
    link_python: bool = True,
) -> None:
    """Compile ``source`` into the shared library ``output``.

    ``libraries``/``library_dirs`` are given as bare names and directories;
    each toolchain spells them its own way.
    """
    # The compiler runs from the output directory, so every path must be absolute.
    source = source.resolve()
    output = output.resolve()
    includes = [str(Path(include).resolve()) for include in includes]
    library_dirs = [str(Path(directory).resolve()) for directory in library_dirs]
    driver = toolchain.driver(cpp=cpp)
    if driver is None:
        raise ToolchainUnavailable(f"no C{'++' if cpp else ''} compiler found")
    # Limited-API builds link the stable import library, not the versioned one.
    limited_api = any(define.startswith("Py_LIMITED_API") for define in defines)
    if toolchain.kind == "msvc":
        cmd = _msvc_command(
            driver=driver,
            source=source,
            output=output,
            cpp=cpp,
            includes=includes,
            defines=defines,
            stdlib=stdlib,
            libraries=libraries,
            library_dirs=library_dirs,
            link_python=link_python,
            limited_api=limited_api,
        )
    else:
        cmd = _gnu_command(
            driver=driver,
            source=source,
            output=output,
            includes=includes,
            defines=defines,
            stdlib=stdlib,
            libraries=libraries,
            library_dirs=library_dirs,
            link_python=link_python,
            limited_api=limited_api,
        )
    env = _merged_environment(toolchain.env)
    # Run inside the output directory: MSVC scatters .obj/.exp/.lib intermediates
    # next to the working directory, which would otherwise litter the repo root.
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env, cwd=output.parent
    )
    if proc.returncode != 0:
        # A present compiler that fails to produce a fixture is a real bug, not
        # an environment issue. Skipping silently masks linker regressions.
        raise RuntimeError(
            f"compile failed (exit {proc.returncode}):\n"
            f"$ {' '.join(cmd)}\n"
            f"{(proc.stderr or proc.stdout).strip()}"
        )


def _merged_environment(overrides: dict[str, str] | None) -> dict[str, str] | None:
    """Process environment with ``overrides`` winning, case-insensitively.

    Windows variable names are case-insensitive but dict keys are not: the
    process environment usually spells it ``Path`` while ``vcvars`` emits
    ``PATH``. Merging naively leaves both in the dict, and the child may then
    see the original ``Path`` — without the MSVC tool directories — instead of
    vcvars' value.
    """
    if not overrides:
        return None
    replaced = {key.upper() for key in overrides}
    env = {
        key: value for key, value in os.environ.items() if key.upper() not in replaced
    }
    env.update(overrides)
    return env


def _msvc_command(
    *,
    driver: str,
    source: Path,
    output: Path,
    cpp: bool,
    includes: Sequence[str],
    defines: Sequence[str],
    stdlib: str | None,
    libraries: Sequence[str],
    library_dirs: Sequence[str],
    link_python: bool,
    limited_api: bool,
) -> list[str]:
    # /MD matches CPython's own CRT; /LD produces a DLL (a .pyd is one).
    cmd = [driver, "/nologo", "/LD", "/MD", "/O2"]
    if cpp:
        cmd.append("/EHsc")
    if stdlib is not None:
        cmd.append(f"/std:{stdlib}")
    cmd.extend(f"/D{define}" for define in defines)
    cmd.extend(f"/I{include}" for include in includes)
    cmd.append(str(source))
    cmd.extend(["/link", f"/OUT:{output}"])
    cmd.extend(f"/LIBPATH:{directory}" for directory in library_dirs)
    # Bare names so a library without a search directory still links, the way
    # -l does in the GNU branch.
    cmd.extend(f"{library}.lib" for library in libraries)
    if link_python:
        import_lib = _python_import_library(limited_api=limited_api)
        # LIBPATH as well: the versioned import library asks the linker for
        # python3.lib by name, which is only resolvable on that directory.
        cmd.extend([f"/LIBPATH:{import_lib.parent}", str(import_lib)])
    return cmd


def _gnu_command(
    *,
    driver: str,
    source: Path,
    output: Path,
    includes: Sequence[str],
    defines: Sequence[str],
    stdlib: str | None,
    libraries: Sequence[str],
    library_dirs: Sequence[str],
    link_python: bool,
    limited_api: bool,
) -> list[str]:
    cmd = [driver, "-shared", "-fPIC", "-O1", "-o", str(output), str(source)]
    cmd.extend(f"-I{include}" for include in includes)
    cmd.extend(f"-D{define}" for define in defines)
    if stdlib is not None:
        cmd.append(f"-std={stdlib}")
    cmd.extend(f"-L{directory}" for directory in library_dirs)
    cmd.extend(f"-l{library}" for library in libraries)
    if sys.platform == "darwin":
        # Mach-O's linker rejects undefined Python C API symbols at link time;
        # defer resolution to load time, which is how setuptools builds Python
        # extensions on macOS.
        cmd.extend(["-undefined", "dynamic_lookup"])
    elif sys.platform == "win32" and link_python:
        # Unlike ELF, PE has no unresolved symbols: link the import library.
        cmd.append(str(_python_import_library(limited_api=limited_api)))
    return cmd
