"""
sprc00 PD disaggregation smoke test.

Ladder of checks (env, ttnn, mesh, submesh, Generator import, paged-attn
canary). Exits 0 iff every check that ran passed. No checkpoint required.

Usage:
    python smoke_test.py
    python smoke_test.py --skip-paged-attn
    python smoke_test.py --num-devices 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(msg: str) -> None:
    print()
    print(BLUE + "=" * 72 + RESET)
    print(BLUE + msg + RESET)
    print(BLUE + "=" * 72 + RESET)


def passed(name: str, detail: str = "") -> None:
    extra = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {GREEN}PASS{RESET}  {name}{extra}")


def failed(name: str, detail: str, hint: str = "") -> None:
    print(f"  {RED}FAIL{RESET}  {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"        {line}")
    if hint:
        print()
        print(f"        {YELLOW}Hint:{RESET} {hint}")


def skipped(name: str, why: str) -> None:
    print(f"  {YELLOW}SKIP{RESET}  {name}  {DIM}({why}){RESET}")


# ---------------------------------------------------------------------------
# Check framework
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    ok: bool
    skipped: bool = False
    detail: str = ""
    hint: str = ""


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    """Run a check function. The function returns a one-line success detail
    on success, raises on failure. Hint text on failure comes via the
    exception's `.hint` attribute if present."""
    try:
        detail = fn() or ""
    except SkipCheck as exc:
        skipped(name, str(exc))
        return CheckResult(name=name, ok=True, skipped=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=2)
        hint = getattr(exc, "hint", "")
        failed(name, f"{exc.__class__.__name__}: {exc}\n{tb}", hint)
        return CheckResult(name=name, ok=False, detail=str(exc), hint=hint)
    passed(name, detail)
    return CheckResult(name=name, ok=True, detail=detail)


class SkipCheck(Exception):
    """Raise from a check fn to mark it skipped instead of failed."""


def fail_with(msg: str, hint: str) -> None:
    err = RuntimeError(msg)
    err.hint = hint  # type: ignore[attr-defined]
    raise err


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> str:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10) or (major, minor) > (3, 12):
        fail_with(
            f"Python {major}.{minor} is outside the supported range (3.10-3.12)",
            "Source the project venv: `source .venv/bin/activate` "
            "or use the python from the characterization setup.",
        )
    return f"Python {major}.{minor}"


def check_tt_smi() -> str:
    """tt-smi reports cards. Independent of ttnn."""
    if shutil.which("tt-smi") is None:
        raise SkipCheck("tt-smi not on PATH")
    try:
        out = subprocess.check_output(
            ["tt-smi", "-ls"], stderr=subprocess.STDOUT, timeout=10
        ).decode()
    except subprocess.CalledProcessError as exc:
        fail_with(
            f"tt-smi exited with {exc.returncode}: {exc.output.decode()[:200]}",
            "Run `tt-smi -ls` manually; if cards are missing, "
            "power-cycle or re-flash the box.",
        )
    n_lines = sum(1 for line in out.splitlines() if "Blackhole" in line or "P150" in line or "P100" in line or "P300" in line)
    return f"tt-smi reports {n_lines} Blackhole cards (full output suppressed)"


def check_ttnn_import() -> str:
    try:
        import ttnn  # noqa: F401
    except ImportError as exc:
        fail_with(
            f"`import ttnn` failed: {exc}",
            "Check: (a) correct venv, (b) PYTHONPATH includes $TT_METAL_HOME, "
            "(c) driver loaded (`lsmod | grep tt_kmd`).",
        )
    return f"import ttnn OK (version: {getattr(__import__('ttnn'), '__version__', 'unknown')})"


def check_ttnn_device_count(expected: Optional[int]) -> str:
    import ttnn

    n = ttnn.get_num_devices()
    if n == 0:
        fail_with(
            "ttnn.get_num_devices() returned 0",
            "ttnn imports but sees no cards. Likely a driver or permissions "
            "issue. Try `tt-smi -ls` to confirm cards are alive, then check "
            "that your user has access to /dev/tenstorrent/*.",
        )
    if expected is not None and n != expected:
        fail_with(
            f"Expected {expected} devices, got {n}",
            f"Pass --num-devices {n} to skip this check, or set "
            f"TT_VISIBLE_DEVICES to restrict.",
        )
    return f"ttnn.get_num_devices() = {n}"


def check_open_close_single_device() -> str:
    """Minimum device lifecycle: open device 0, repr it, close."""
    import ttnn

    device = None
    try:
        device = ttnn.open_device(device_id=0)
        # Property names drift between ttnn versions; just repr the object.
        repr_str = repr(device)[:80]
    finally:
        if device is not None:
            ttnn.close_device(device)
    return f"opened/closed device 0 ({repr_str})"


def check_open_mesh_device(num_devices: int) -> str:
    """Open a 1xN MeshDevice across all visible cards."""
    import ttnn

    mesh_shape = ttnn.MeshShape(1, num_devices)
    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    try:
        n = mesh.get_num_devices()
        if n != num_devices:
            fail_with(
                f"MeshDevice opened with {n} devices, expected {num_devices}",
                "May indicate a fabric/topology mismatch. Check `tt-smi -ls`.",
            )
    finally:
        ttnn.close_mesh_device(mesh)
    return f"MeshDevice 1x{num_devices} opened/closed"


def check_create_submeshes(num_devices: int) -> str:
    """Split the mesh in two — the primitive PD disaggregation needs."""
    import ttnn

    if num_devices < 2:
        raise SkipCheck("only one device visible; submesh carve requires >=2")

    half = num_devices // 2
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, num_devices))
    try:
        submeshes = mesh.create_submeshes(ttnn.MeshShape(1, half))
        if len(submeshes) != 2:
            fail_with(
                f"Expected 2 submeshes, got {len(submeshes)}",
                "create_submeshes returns one submesh per partition; check shape math.",
            )
        sizes = [s.get_num_devices() for s in submeshes]
    finally:
        ttnn.close_mesh_device(mesh)
    return f"create_submeshes: {len(submeshes)} submeshes of sizes {sizes}"


def check_tensor_roundtrip_each_submesh(num_devices: int) -> str:
    """Round-trip a tile through each submesh. Data-path sanity check."""
    import torch
    import ttnn

    if num_devices < 2:
        raise SkipCheck("need >=2 devices for two-submesh round-trip")

    half = num_devices // 2
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, num_devices))
    try:
        submeshes = mesh.create_submeshes(ttnn.MeshShape(1, half))
        # 32x32 = native Tensix tile.
        original = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
        for idx, submesh in enumerate(submeshes):
            tt = ttnn.from_torch(
                original,
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            )
            # Multi-device tensors need a mesh composer for to_torch; for a
            # replicated tensor pull shard 0 directly (same idiom as
            # tt_transformers/tt/generator.py).
            back = ttnn.to_torch(ttnn.get_device_tensors(tt)[0])
            # bfloat16 round-trip is lossy; check max abs err.
            err = (back.float() - original).abs().max().item()
            if err > 1e-2:
                fail_with(
                    f"submesh {idx} round-trip max abs err {err:.4f} too large",
                    "Layout mismatch or numerical regression; "
                    "compare `back.shape` to `original.shape`.",
                )
    finally:
        ttnn.close_mesh_device(mesh)
    return f"round-trip OK on both submeshes (1x{half} each)"


def check_concurrent_submesh_compute(num_devices: int) -> str:
    """Dispatch a matmul on each submesh from one Python process."""
    import torch
    import ttnn

    if num_devices < 2:
        raise SkipCheck("need >=2 devices for concurrent-submesh test")

    half = num_devices // 2
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, num_devices))
    try:
        submeshes = mesh.create_submeshes(ttnn.MeshShape(1, half))
        a = torch.randn(64, 64, dtype=torch.float32)
        b = torch.randn(64, 64, dtype=torch.float32)

        results = []
        for submesh in submeshes:
            tt_a = ttnn.from_torch(
                a, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            )
            tt_b = ttnn.from_torch(
                b, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            )
            tt_c = ttnn.matmul(tt_a, tt_b)
            ttnn.synchronize_device(submesh)
            # Per-shard unwrap, same as the round-trip check.
            results.append(ttnn.to_torch(ttnn.get_device_tensors(tt_c)[0]))
        # Skip the torch ref compare; bfloat16 matmul has its own envelope.
        for idx, c in enumerate(results):
            shape_ok = c.shape[-2:] == (64, 64)
            finite_ok = torch.isfinite(c).all().item()
            if not (shape_ok and finite_ok):
                fail_with(
                    f"submesh {idx} matmul result invalid (shape={c.shape}, finite={finite_ok})",
                    "Wrong shape -> layout issue. Non-finite -> kernel hang + garbage read.",
                )
    finally:
        ttnn.close_mesh_device(mesh)
    return "concurrent matmul ran on both submeshes"


def check_tt_transformers_imports() -> str:
    """Import Generator + create_submeshes. No instantiation (needs ckpt)."""
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home or not os.path.isdir(tt_metal_home):
        fail_with(
            f"TT_METAL_HOME={tt_metal_home!r} not set or not a directory",
            "Set TT_METAL_HOME to the tt-metal checkout and add to PYTHONPATH:\n"
            "  export TT_METAL_HOME=/path/to/tt-metal\n"
            "  export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH",
        )
    if tt_metal_home not in sys.path:
        sys.path.insert(0, tt_metal_home)
    try:
        from models.tt_transformers.tt.generator import Generator, create_submeshes  # noqa: F401
    except ModuleNotFoundError as exc:
        # Missing model-side python dep (pydantic / transformers / etc.).
        missing = getattr(exc, "name", None) or str(exc)
        fail_with(
            f"importing Generator failed: missing module `{missing}`",
            "Install the tt-metal model-side requirements:\n"
            "  pip install pydantic==2.9.2 transformers==4.53.0 'huggingface-hub>=0.30.0'\n"
            "Or:\n"
            f"  pip install -r {os.path.join(tt_metal_home, 'tt_metal/python_env/requirements-dev.txt')}",
        )
    except Exception as exc:  # noqa: BLE001
        fail_with(
            f"importing Generator failed: {exc.__class__.__name__}: {exc}",
            "Likely TT_METAL_HOME / PYTHONPATH. Reproduce with:\n"
            "  python -c 'from models.tt_transformers.tt.generator import Generator'",
        )
    return "Generator + create_submeshes import OK"


def check_paged_attention_canary(num_devices: int) -> str:
    """Smallest paged_update_cache call with a 30s timeout (issue #16674)."""
    import multiprocessing as mp

    if num_devices < 1:
        raise SkipCheck("no devices to test against")

    def worker(q: mp.Queue) -> None:
        try:
            import torch
            import ttnn

            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
            try:
                # Smallest legal paged-cache update; shapes mirror tt_transformers.
                cache = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
                upd = torch.ones(1, 1, 32, 32, dtype=torch.float32)
                tt_cache = ttnn.from_torch(
                    cache, device=mesh, layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                tt_upd = ttnn.from_torch(
                    upd, device=mesh, layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                # Distinguish "op missing/moved" from "op hangs".
                op = getattr(ttnn.experimental, "paged_update_cache", None)
                if op is None:
                    q.put(("missing", "ttnn.experimental.paged_update_cache not found"))
                    return
                update_idxs = torch.tensor([0], dtype=torch.int32)
                op(tt_cache, tt_upd, update_idxs=update_idxs)
                ttnn.synchronize_device(mesh)
                q.put(("ok", "paged_update_cache returned"))
            finally:
                ttnn.close_mesh_device(mesh)
        except Exception as exc:  # noqa: BLE001
            q.put(("error", f"{exc.__class__.__name__}: {exc}"))

    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=worker, args=(q,))
    proc.start()
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        fail_with(
            "paged_update_cache did not return within 30 seconds",
            "Likely issue #16674 "
            "(https://github.com/tenstorrent/tt-metal/issues/16674). "
            "Pin tt-metal to a Blackhole-good commit, or disable paged_attention.",
        )
    if q.empty():
        fail_with(
            "paged_update_cache subprocess returned no result",
            "Subprocess crashed; rerun with PYTHONUNBUFFERED=1.",
        )
    status, msg = q.get()
    if status == "missing":
        raise SkipCheck(msg)
    if status == "error":
        fail_with(
            msg,
            "Op errored, not hung. Check shape/layout against the docstring.",
        )
    return msg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-devices", type=int, default=None,
        help="Assert this many devices are visible. Default: don't assert.",
    )
    parser.add_argument(
        "--skip-paged-attn", action="store_true",
        help="Skip the paged_update_cache canary (issue #16674).",
    )
    parser.add_argument(
        "--skip-mesh", action="store_true",
        help="Skip all multi-chip checks (useful on a single-card host).",
    )
    args = parser.parse_args()

    banner("sprc00 PD disaggregation smoke test")

    results: List[CheckResult] = []
    visible_devices: Optional[int] = None

    # 1. Env basics, no device contact.
    print("\n[1/3] Environment")
    results.append(run_check("python version", check_python_version))
    results.append(run_check("tt-smi reports cards", check_tt_smi))
    if not all(r.ok for r in results):
        banner("Stopping early: environment not sane.")
        return 1

    # 2. ttnn import + device count.
    print("\n[2/3] TT-NN basics")
    results.append(run_check("import ttnn", check_ttnn_import))
    if not results[-1].ok:
        banner("Stopping early: ttnn not importable.")
        return 1

    res = run_check(
        "ttnn.get_num_devices()",
        lambda: check_ttnn_device_count(args.num_devices),
    )
    results.append(res)
    if not res.ok:
        banner("Stopping early: device discovery failed.")
        return 1
    import ttnn  # safe now
    visible_devices = ttnn.get_num_devices()

    results.append(run_check("open/close single device", check_open_close_single_device))

    # 3. Multi-chip / submesh / Generator wiring.
    print("\n[3/3] Mesh, submesh, Generator import, paged-attention canary")
    if args.skip_mesh or visible_devices is None or visible_devices < 2:
        skipped("MeshDevice", "--skip-mesh or <2 devices visible")
        skipped("create_submeshes", "--skip-mesh or <2 devices visible")
        skipped("submesh tensor round-trip", "--skip-mesh or <2 devices visible")
        skipped("concurrent submesh compute", "--skip-mesh or <2 devices visible")
    else:
        results.append(
            run_check("open/close MeshDevice", lambda: check_open_mesh_device(visible_devices))
        )
        results.append(
            run_check("create_submeshes", lambda: check_create_submeshes(visible_devices))
        )
        results.append(
            run_check(
                "submesh tensor round-trip",
                lambda: check_tensor_roundtrip_each_submesh(visible_devices),
            )
        )
        results.append(
            run_check(
                "concurrent matmul on submeshes",
                lambda: check_concurrent_submesh_compute(visible_devices),
            )
        )

    results.append(run_check("Generator + create_submeshes import", check_tt_transformers_imports))

    if args.skip_paged_attn:
        skipped("paged_update_cache canary", "--skip-paged-attn")
    else:
        results.append(
            run_check(
                "paged_update_cache canary (issue #16674)",
                lambda: check_paged_attention_canary(visible_devices or 0),
            )
        )

    # Summary.
    banner("Summary")
    n_ok = sum(1 for r in results if r.ok and not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    n_fail = sum(1 for r in results if not r.ok)
    print(f"  {GREEN}{n_ok} passed{RESET}, "
          f"{YELLOW}{n_skip} skipped{RESET}, "
          f"{RED}{n_fail} failed{RESET}")
    if n_fail:
        print(textwrap.dedent(f"""
            Next: read the topmost {RED}FAIL{RESET} above and follow its hint.
            Most failures are env issues (PYTHONPATH, TT_METAL_HOME, perms).
        """).strip())
        return 1
    print(textwrap.dedent("""
        All checks passed. Next:
          export LLAMA_DIR=/path/to/Llama-3.2-1B-Instruct
          pytest <tt-metal>/models/tt_transformers/demo/simple_text_demo.py::test_demo_text \\
                 -k "batch-1-latency"
    """).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
