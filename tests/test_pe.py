"""PE/COFF symbol extraction tests.

The images are synthesised in-process (:mod:`pe_builder`), so these run on every
OS and need no compiler. Real-world behaviour is covered by the module's own
docs and by auditing actual Windows wheels in CI.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import pytest

from pe_builder import PEBuilder, write_pe
from torch_abi_audit import inspect_package, pe
from torch_abi_audit.pe import PEFormatError, read_pe
from torch_abi_audit.symbols import (
    extract_undefined_symbols,
    find_compiled_libraries,
    is_extension_module,
    read_symbols,
)

# MSVC-mangled references, as found in the import tables of real Windows wheels.
AT_TENSOR_NUMEL = "?numel@Tensor@at@@QEBA_JXZ"
STABLE_TENSOR_DIM = "?dim@Tensor@stable@torch@@QEBA_JXZ"
AOTI_GET_DIM = "aoti_torch_get_dim"
CPYTHON_UNSTABLE = "_PyDict_NewPresized"
CPYTHON_STABLE = "PyModule_Create2"


def test_reads_imports_and_exports(tmp_path: Path) -> None:
    image = write_pe(
        tmp_path / "demo.pyd",
        PEBuilder()
        .add_export("PyInit_demo")
        .add_export("PyModExport_demo")
        .add_import("python314.dll", CPYTHON_STABLE)
        .add_import("torch_cpu.dll", AOTI_GET_DIM),
    )

    imports, exports = read_pe(image)

    assert exports == ("PyInit_demo", "PyModExport_demo")
    # Group order follows the import descriptors; entries keep table order.
    assert imports == (CPYTHON_STABLE, AOTI_GET_DIM)


def test_ordinal_only_imports_are_skipped(tmp_path: Path) -> None:
    image = write_pe(
        tmp_path / "demo.dll",
        PEBuilder()
        .add_import("python314.dll", CPYTHON_STABLE)
        .add_import("python314.dll", None),
    )

    imports, exports = read_pe(image)

    assert imports == (CPYTHON_STABLE,)
    assert exports == ()


@pytest.mark.parametrize(
    ("pe32_plus", "rva_based"),
    [(True, True), (False, True), (False, False)],
)
def test_delay_imports(tmp_path: Path, pe32_plus: bool, rva_based: bool) -> None:
    """Both delay-load descriptor flavours: RVAs since VS6, plain VAs before.

    The VA form only exists for 32-bit images — descriptor fields are DWORDs, so
    a 64-bit image base cannot be stored in them.
    """
    image = write_pe(
        tmp_path / "demo.pyd",
        PEBuilder(pe32_plus=pe32_plus).add_delay_import(
            "torch_cpu.dll", AOTI_GET_DIM, rva_based=rva_based
        ),
    )

    imports, _ = read_pe(image)

    assert imports == (AOTI_GET_DIM,)


@pytest.mark.parametrize("pe32_plus", [True, False])
def test_both_optional_header_formats(tmp_path: Path, pe32_plus: bool) -> None:
    image = write_pe(
        tmp_path / ("demo64.pyd" if pe32_plus else "demo32.pyd"),
        PEBuilder(pe32_plus=pe32_plus)
        .add_export("PyInit_demo")
        .add_import("torch_cpu.dll", AOTI_GET_DIM),
    )

    imports, exports = read_pe(image)

    assert exports == ("PyInit_demo",)
    assert imports == (AOTI_GET_DIM,)


def test_image_without_tables(tmp_path: Path) -> None:
    image = write_pe(tmp_path / "empty.dll", PEBuilder())

    assert read_pe(image) == ((), ())


def test_rejects_non_pe(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.dll"
    bogus.write_bytes(b"not a pe image at all")

    with pytest.raises(PEFormatError):
        read_pe(bogus)


def test_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pyd"
    empty.touch()

    with pytest.raises(PEFormatError):
        read_pe(empty)


def test_rejects_truncated_image(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.pyd"
    truncated.write_bytes(PEBuilder().add_export("PyInit_demo").build()[:0x50])

    with pytest.raises(PEFormatError):
        read_pe(truncated)


def test_unparseable_pe_yields_no_symbols(tmp_path: Path) -> None:
    """`read_symbols` mirrors nm's leniency instead of raising."""
    broken = tmp_path / "broken.dll"
    broken.write_bytes(b"MZ" + b"\x00" * 0x3A + b"\xff\xff\xff\x7f")

    assert read_symbols(broken) == read_symbols(tmp_path / "does-not-exist.dll")


def test_read_symbols_demangles_msvc_imports(tmp_path: Path) -> None:
    image = write_pe(
        tmp_path / "demo.pyd",
        PEBuilder()
        .add_export("PyInit_demo")
        .add_import("torch_cpu.dll", AT_TENSOR_NUMEL)
        .add_import("torch_cpu.dll", STABLE_TENSOR_DIM),
    )

    symbols = read_symbols(image)

    # Exports stay raw: entry-point detection only needs the plain C name.
    assert symbols.exports == ("PyInit_demo",)
    assert any("at::Tensor::numel" in sym for sym in symbols.imports)
    assert any("torch::stable::Tensor::dim" in sym for sym in symbols.imports)
    # The public wrapper used by inspect.py takes the same route.
    assert extract_undefined_symbols(image) == list(symbols.imports)


def test_extension_detection_uses_export_table(tmp_path: Path) -> None:
    extension = write_pe(tmp_path / "demo.pyd", PEBuilder().add_export("PyInit_demo"))
    plugin = write_pe(
        tmp_path / "plugin.abi3.dll", PEBuilder().add_export("stable_torch_plugin")
    )

    assert is_extension_module(extension) is True
    # An abi3 filename alone is not enough — torch plugins use it too.
    assert is_extension_module(plugin) is False


def test_find_compiled_libraries_includes_dlls(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    write_pe(pkg / "a.dll", PEBuilder())
    write_pe(pkg / "b.pyd", PEBuilder())
    # Linker import libraries are archives, not PE images; they must not match.
    (pkg / "c.lib").write_bytes(b"!<arch>\n")

    assert {p.name for p in find_compiled_libraries(pkg)} == {"a.dll", "b.pyd"}


# The builder lays the single section out right after the 0x200-byte headers and
# reserves the export directory first, so these are stable offsets into its data.
_SECTION_RAW = 0x200
_EXPORT_NUMBER_OF_NAMES = _SECTION_RAW + 24
_EXPORT_ADDRESS_OF_NAMES = _SECTION_RAW + 32


def _patch_image(path: Path, offset: int, value: int) -> None:
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, offset, value)
    path.write_bytes(bytes(data))


def test_implausible_export_count_is_rejected(tmp_path: Path) -> None:
    """The count comes from the image, so it is checked before anything is
    allocated — ``read_pe`` then degrades to "no exports" instead of raising."""
    image = write_pe(
        tmp_path / "demo.pyd",
        PEBuilder().add_export("PyInit_demo").add_import("torch_cpu.dll", AOTI_GET_DIM),
    )
    _patch_image(image, _EXPORT_NUMBER_OF_NAMES, 0xFFFFFFFF)

    with image.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        with pytest.raises(PEFormatError):
            pe._Image(m).exports()

    imports, exports = read_pe(image)
    assert exports == ()
    assert imports == (AOTI_GET_DIM,)


def test_broken_export_directory_keeps_imports(tmp_path: Path) -> None:
    """A damaged export table must not throw the import table away — that would
    report a torch-using library as not using torch at all."""
    image = write_pe(
        tmp_path / "demo.pyd",
        PEBuilder().add_export("PyInit_demo").add_import("torch_cpu.dll", AOTI_GET_DIM),
    )
    # Name-pointer array points outside every section.
    _patch_image(image, _EXPORT_ADDRESS_OF_NAMES, 0xDEAD000)

    imports, exports = read_pe(image)

    assert imports == (AOTI_GET_DIM,)
    assert exports == ()


def test_abi3_tagged_pe_is_compliant(tmp_path: Path) -> None:
    image = write_pe(
        tmp_path / "demo.abi3.pyd",
        PEBuilder()
        .add_export("PyInit_demo")
        .add_import("python314.dll", CPYTHON_STABLE),
    )

    report = inspect_package(image)
    extension = report.extensions[0]

    assert extension.cpython.intent is True
    assert extension.cpython.compliant is True
    assert report.cpython_verdict == "abi3-compliant"


def test_inspect_package_classifies_windows_libraries(tmp_path: Path) -> None:
    write_pe(
        tmp_path / "_demo.pyd",
        PEBuilder()
        .add_export("PyInit_demo")
        .add_import("python314.dll", CPYTHON_UNSTABLE)
        .add_import("torch_cpu.dll", AT_TENSOR_NUMEL),
    )
    write_pe(
        tmp_path / "libaoti.dll",
        PEBuilder().add_import("torch_cpu.dll", AOTI_GET_DIM),
    )

    report = inspect_package(tmp_path)

    assert [e.path.name for e in report.extensions] == ["_demo.pyd"]
    assert [b.path.name for b in report.bundled_libs] == ["libaoti.dll"]
    assert report.torch_verdict == "torch-unstable"
    assert report.cpython_verdict == "not-abi3"
    assert report.min_torch_version == "2.9.0"
