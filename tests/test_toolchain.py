"""Tests for the fixture toolchain: command construction and environment handling."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import toolchain
from toolchain import (
    _gnu_command,
    _merged_environment,
    _msvc_command,
    _parse_env_block,
    _vcvars_environment,
    extension_suffix,
    find_toolchain,
)


def _msvc(
    *,
    libraries: tuple[str, ...] = (),
    library_dirs: tuple[str, ...] = (),
) -> list[str]:
    return _msvc_command(
        driver="cl.exe",
        source=Path("s.c"),
        output=Path("o.pyd"),
        cpp=False,
        includes=(),
        defines=(),
        stdlib=None,
        libraries=libraries,
        library_dirs=library_dirs,
        link_python=False,
        limited_api=False,
    )


def test_msvc_libraries_do_not_require_a_search_directory() -> None:
    """``libraries`` and ``library_dirs`` are orthogonal, as in the GNU branch.

    Pairing them as a cross product silently dropped every library that had no
    explicit directory, producing a weaker link that only fails later, e.g. as
    unresolved externals.
    """
    cmd = _msvc(libraries=("torch_cpu",))
    assert "torch_cpu.lib" in cmd
    assert not any(arg.startswith("/LIBPATH:") for arg in cmd)

    cmd = _msvc(libraries=("torch_cpu",), library_dirs=("C:/torch/lib",))
    assert "/LIBPATH:C:/torch/lib" in cmd
    assert "torch_cpu.lib" in cmd


def test_gnu_libraries_and_directories_are_separate_flags() -> None:
    cmd = _gnu_command(
        driver="cc",
        source=Path("s.c"),
        output=Path("o.so"),
        includes=("C:/inc",),
        defines=("FOO=1",),
        stdlib=None,
        libraries=("torch_cpu",),
        library_dirs=("C:/torch/lib",),
        link_python=False,
        limited_api=False,
    )
    assert "-LC:/torch/lib" in cmd
    assert "-ltorch_cpu" in cmd
    assert "-IC:/inc" in cmd
    assert "-DFOO=1" in cmd


def test_parse_env_block_skips_pseudo_variables() -> None:
    parsed = _parse_env_block(
        "Path=C:\\Windows\r\n=C:=C:\\repo\r\nINCLUDE=C:\\inc\r\n\r\nPROMPT=$P$G\r\n"
    )
    assert parsed is not None
    assert parsed["INCLUDE"] == "C:\\inc"
    assert parsed["Path"] == "C:\\Windows"
    assert "=C:" not in parsed


def test_parse_env_block_returns_none_when_empty() -> None:
    assert _parse_env_block("\r\n\r\n") is None


def test_vcvars_output_is_decoded_as_oem(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd.exe builtins write to a pipe in the OEM code page.

    Decoding with the ANSI code page turns a profile such as ``C:\\Users\\José``
    into U+FFFD, which then breaks INCLUDE/LIB far from the cause.
    """
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess([], 0, stdout="PATH=C:\\Tools\r\n")

    monkeypatch.setattr(toolchain.subprocess, "run", fake_run)

    env = _vcvars_environment(Path("vcvars64.bat"))

    assert env == {"PATH": "C:\\Tools"}
    assert seen["encoding"] == "oem"


def test_vcvars_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, stdout=""),
    )
    assert _vcvars_environment(Path("vcvars64.bat")) is None


def test_environment_merge_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows spells the variable ``Path`` while vcvars emits ``PATH``.

    Keeping both names in the dict lets the child see the version without the
    MSVC tool directories, making compiler discovery depend on dict ordering.
    """
    # A plain dict rather than os.environ: on Windows os.environ normalises key
    # casing, so it cannot express the collision this guards against.
    monkeypatch.setattr(
        toolchain,
        "os",
        types.SimpleNamespace(environ={"Path": "C:\\Windows", "KEEP": "1"}),
    )

    env = _merged_environment({"PATH": "C:\\Tools;C:\\Windows", "INCLUDE": "C:\\inc"})

    assert env is not None
    assert [key.upper() for key in env].count("PATH") == 1
    assert env["PATH"] == "C:\\Tools;C:\\Windows"
    assert "Path" not in env
    assert env["KEEP"] == "1"
    assert env["INCLUDE"] == "C:\\inc"


def test_no_overrides_means_inherit() -> None:
    assert _merged_environment(None) is None


def test_extension_suffix_matches_the_platform() -> None:
    assert extension_suffix() == (".pyd" if sys.platform == "win32" else ".so")


def test_ci_requires_a_discoverable_toolchain() -> None:
    """Guards against a regression that turns every fixture test into a skip."""
    if not os.environ.get("TORCH_ABI_AUDIT_REQUIRE_TOOLCHAIN"):
        pytest.skip("set TORCH_ABI_AUDIT_REQUIRE_TOOLCHAIN=1 to require a compiler")
    assert find_toolchain() is not None
