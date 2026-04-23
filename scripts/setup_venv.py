#!/usr/bin/env python3
"""Bootstrap a pupa_counter_v6 venv with the right torch wheel for
this machine. Run ONCE per new machine — does the GPU-detection work
that plain ``pip install -r requirements.txt`` can't do.

Detection ladder:
  macOS (any arch)        → default wheel          (MPS on Apple Silicon)
  Windows + NVIDIA GPU    → cu126 wheel
  Windows + Intel Arc GPU → xpu wheel              (torch 2.5+)
  Windows + other GPU     → default + torch-directml
  Linux + NVIDIA GPU      → cu126 wheel
  everything else         → default (CPU only)

Usage:
    python scripts/setup_venv.py
    python scripts/setup_venv.py --venv .venv       # custom venv dir
    python scripts/setup_venv.py --dry-run          # print plan, install nothing
    python scripts/setup_venv.py --force-cpu        # skip GPU detection
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
REQS = REPO / "requirements.txt"


def list_windows_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object -ExpandProperty Name"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def list_linux_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "lspci 2>/dev/null | grep -Ei 'vga|3d|display'"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def detect_plan(force_cpu: bool = False) -> tuple[str, str | None, list[str]]:
    """Return (label, pip_index_url_or_None, extra_pkgs)."""
    if force_cpu:
        return ("CPU (forced)", None, [])

    system = platform.system()

    if system == "Darwin":
        # default wheel has MPS for Apple Silicon and CPU otherwise
        return ("macOS (MPS on Apple Silicon, CPU on Intel Macs)", None, [])

    if system == "Windows":
        gpus = list_windows_gpus()
        joined = " | ".join(gpus).lower()
        if any(k in joined for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
            return (f"Windows + NVIDIA GPU ({gpus[0] if gpus else 'unknown'})",
                    "https://download.pytorch.org/whl/cu126", [])
        # Intel GPUs: Arc discrete, Arc iGPU in Core Ultra (140T etc.), Iris Xe
        if any(k in joined for k in ("intel(r) arc", "intel arc", "iris xe", "intel graphics")):
            return (f"Windows + Intel GPU ({gpus[0] if gpus else 'unknown'})",
                    "https://download.pytorch.org/whl/xpu", [])
        # AMD or unrecognized — default torch + torch-directml gives GPU fallback
        if any(k in joined for k in ("amd", "radeon", "rx ")):
            return (f"Windows + AMD GPU ({gpus[0] if gpus else 'unknown'}), via DirectML",
                    None, ["torch-directml"])
        return (f"Windows, no recognized GPU ({gpus[0] if gpus else 'unknown'}) — CPU",
                None, [])

    if system == "Linux":
        gpus = list_linux_gpus()
        joined = " | ".join(gpus).lower()
        if "nvidia" in joined:
            return (f"Linux + NVIDIA GPU",
                    "https://download.pytorch.org/whl/cu126", [])
        return ("Linux, no recognized GPU — CPU", None, [])

    return (f"Unknown platform ({system}) — CPU", None, [])


def run(cmd: list[str], dry: bool) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=str(REPO / ".venv"),
                    help="Venv directory to create/use (default: .venv at repo root).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the detected plan and pip commands, but don't run them.")
    ap.add_argument("--force-cpu", action="store_true",
                    help="Skip GPU detection and install the default CPU wheel.")
    args = ap.parse_args()

    label, index_url, extras = detect_plan(force_cpu=args.force_cpu)
    print(f"Detected: {label}")
    print(f"  torch index : {index_url or '(default)'}")
    print(f"  extras      : {', '.join(extras) if extras else '(none)'}")
    print()

    venv_path = Path(args.venv).resolve()
    if venv_path.exists():
        print(f"Venv already exists at {venv_path} — will install into it.")
    else:
        print(f"Creating venv at {venv_path}")
        if not args.dry_run:
            venv.EnvBuilder(with_pip=True, symlinks=(platform.system() != "Windows")).create(str(venv_path))

    if platform.system() == "Windows":
        py = venv_path / "Scripts" / "python.exe"
    else:
        py = venv_path / "bin" / "python"

    if not args.dry_run and not py.exists():
        print(f"ERROR: expected venv python at {py} but it's missing", file=sys.stderr)
        return 1

    print()
    print("Upgrading pip...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)

    print()
    print("Installing torch + torchvision...")
    cmd = [str(py), "-m", "pip", "install", "torch", "torchvision"]
    if index_url:
        cmd += ["--index-url", index_url]
    run(cmd, args.dry_run)

    if extras:
        print()
        print("Installing extras...")
        run([str(py), "-m", "pip", "install", *extras], args.dry_run)

    print()
    print("Installing pinned non-torch deps from requirements.txt...")
    # Strip torch/torchvision from requirements so pip doesn't re-resolve
    # them off the default index.
    filtered = []
    for ln in REQS.read_text(encoding="utf-8").splitlines():
        base = ln.split("#", 1)[0].strip()
        if not base:
            continue
        if base.split(">=")[0].split("==")[0].strip().lower() in ("torch", "torchvision"):
            continue
        filtered.append(base)
    # scikit-learn pin is needed by peak_filter_clf.pkl (sklearn 1.6.x)
    if not any(p.lower().startswith("scikit-learn") for p in filtered):
        filtered.append("scikit-learn==1.6.1")
    if filtered:
        run([str(py), "-m", "pip", "install", *filtered], args.dry_run)

    print()
    print("Verifying torch device...")
    check = (
        "import torch; "
        "has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available(); "
        "has_cuda = torch.cuda.is_available(); "
        "has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False; "
        "name = torch.cuda.get_device_name(0) if has_cuda else torch.xpu.get_device_name(0) if has_xpu else 'CPU'; "
        "print(f'  torch     : {torch.__version__}'); "
        "print(f'  cuda      : {has_cuda}'); "
        "print(f'  mps       : {has_mps}'); "
        "print(f'  xpu       : {has_xpu}'); "
        "print(f'  device    : {name}')"
    )
    run([str(py), "-c", check], args.dry_run)

    print()
    print("Done. Run the daemon with:")
    print(f"  {py} pupa_counter_daemon.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
