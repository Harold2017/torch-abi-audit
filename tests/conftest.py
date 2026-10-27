"""Shared pytest fixtures.

The fixture-builder logic compiles tiny C/C++ extensions on demand against the
running interpreter (and against ``torch`` when available), using MSVC on
Windows and a GNU-style driver elsewhere (see :mod:`toolchain`). If no compiler
is found, the fixtures skip cleanly so the rest of the suite still runs.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

import pytest

from toolchain import (
    Toolchain,
    ToolchainUnavailable,
    build_extension,
    extension_suffix,
    find_toolchain,
)

_SRC = Path(__file__).parent / "fixtures" / "src"

#: CI sets this: every runner has a compiler, so skipping means detection broke.
_REQUIRE_TOOLCHAIN = "TORCH_ABI_AUDIT_REQUIRE_TOOLCHAIN"


def require_toolchain() -> Toolchain:
    """:func:`toolchain.find_toolchain`, skipping the test when there is none."""
    found = find_toolchain()
    if found is None:
        message = "no C/C++ compiler found"
        if os.environ.get(_REQUIRE_TOOLCHAIN):
            pytest.fail(f"{message} ({_REQUIRE_TOOLCHAIN} is set)")
        pytest.skip(message)
    return found


def build_fixture(
    *,
    source: Path,
    output: Path,
    cpp: bool = False,
    includes: tuple[str, ...] = (),
    defines: tuple[str, ...] = (),
    stdlib: str | None = None,
    libraries: tuple[str, ...] = (),
    library_dirs: tuple[str, ...] = (),
    link_python: bool = True,
) -> Path:
    """Compile one fixture, skipping when the needed compiler is missing."""
    toolchain = require_toolchain()
    try:
        build_extension(
            toolchain=toolchain,
            source=source,
            output=output,
            cpp=cpp,
            includes=includes,
            defines=defines,
            stdlib=stdlib,
            libraries=libraries,
            library_dirs=library_dirs,
            link_python=link_python,
        )
    except ToolchainUnavailable as exc:
        pytest.skip(str(exc))
    return output


@pytest.fixture(scope="session")
def cpython_stable_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    suffix = extension_suffix()
    out = tmp_path_factory.mktemp("ext-stable") / f"fixture_stable.abi3{suffix}"
    return build_fixture(
        source=_SRC / "cpython_stable.c",
        output=out,
        includes=(sysconfig.get_paths()["include"],),
        defines=("Py_LIMITED_API=0x030B0000",),
    )


@pytest.fixture(scope="session")
def cpython_unstable_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ext-unstable") / f"fixture_unstable{extension_suffix()}"
    return build_fixture(
        source=_SRC / "cpython_unstable.c",
        output=out,
        includes=(sysconfig.get_paths()["include"],),
    )


@pytest.fixture(scope="session")
def cpython_pymodexport_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.version_info < (3, 15):
        pytest.skip("PyModExport requires Python 3.15+")
    out = tmp_path_factory.mktemp("ext-pymodexport") / (
        f"fixture_pymodexport{extension_suffix()}"
    )
    return build_fixture(
        source=_SRC / "cpython_pymodexport.c",
        output=out,
        includes=(sysconfig.get_paths()["include"],),
    )


def _torch_paths() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    """``(includes, library_dirs, libraries)`` for linking against torch."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    base = Path(torch.__file__).parent
    includes = (
        str(base / "include"),
        str(base / "include" / "torch" / "csrc" / "api" / "include"),
    )
    return includes, (str(base / "lib"),), ("torch", "torch_cpu", "c10")


@pytest.fixture(scope="session")
def torch_stable_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    paths = _torch_paths()
    if paths is None:
        pytest.skip("torch is not importable")
    includes, library_dirs, libraries = paths
    out = tmp_path_factory.mktemp("ext-torch-stable") / (
        f"fixture_torch_stable{extension_suffix()}"
    )
    return build_fixture(
        source=_SRC / "torch_stable.cpp",
        output=out,
        cpp=True,
        includes=includes,
        defines=("TORCH_TARGET_VERSION=0x020a000000000000",),
        stdlib="c++20",
        libraries=libraries,
        library_dirs=library_dirs,
    )


@pytest.fixture(scope="session")
def torch_unstable_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    paths = _torch_paths()
    if paths is None:
        pytest.skip("torch is not importable")
    includes, library_dirs, libraries = paths
    out = tmp_path_factory.mktemp("ext-torch-unstable") / (
        f"fixture_torch_unstable{extension_suffix()}"
    )
    return build_fixture(
        source=_SRC / "torch_unstable.cpp",
        output=out,
        cpp=True,
        includes=includes,
        stdlib="c++20",
        libraries=libraries,
        library_dirs=library_dirs,
    )
