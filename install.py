#!/usr/bin/env python3
"""
OPAC Install Script
====================
Run this once on a fresh machine (Windows or Linux).
It installs all dependencies and verifies your NPU is accessible.

Usage:
    python install.py
    python install.py --no-npu-check    (skip NPU verification)
    python install.py --cpu-only        (use CPU instead of NPU)
"""

import subprocess
import sys
import platform
import argparse
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"
PY         = sys.executable


def run(cmd: list, check=True):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-npu-check", action="store_true")
    parser.add_argument("--cpu-only",     action="store_true")
    args = parser.parse_args()

    print(r"""
  ___  ____   _    ____
 / _ \|  _ \ / \  / ___|
| | | | |_) / _ \| |
| |_| |  __/ ___ \ |___
 \___/|_| /_/   \_\____|

 OPAC Installer
""")
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Python   : {sys.version.split()[0]}")

    # ── Step 1: pip requirements ──────────────────────────────────────────────
    section("Step 1: Installing Python dependencies")
    run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    run([PY, "-m", "pip", "install", "-r", "requirements.txt"])

    # ── Step 2: OpenVINO ─────────────────────────────────────────────────────
    section("Step 2: Installing OpenVINO + GenAI")
    run([PY, "-m", "pip", "install", "openvino", "openvino-genai", "huggingface_hub"])

    # ── Step 3: NPU driver check ──────────────────────────────────────────────
    if not args.no_npu_check:
        section("Step 3: Checking Intel NPU driver")
        try:
            import openvino as ov
            core    = ov.Core()
            devices = core.available_devices
            if "NPU" in devices:
                print(f"  Intel NPU detected  ✓  (devices: {', '.join(devices)})")
            else:
                print(f"  NPU NOT found. Available: {', '.join(devices)}")
                print()
                if IS_WINDOWS:
                    print("  Install NPU driver for Windows:")
                    print("  https://www.intel.com/content/www/us/en/download/794734/")
                    print("  intel-npu-acceleration-library-installer.exe")
                elif IS_LINUX:
                    print("  Install NPU driver for Linux:")
                    print("  https://github.com/intel/linux-npu-driver/releases")
                    print()
                    print("  Quick install (Ubuntu 22.04 / 24.04):")
                    print("    wget https://github.com/intel/linux-npu-driver/releases/latest/download/intel-driver-compiler-npu_*.deb")
                    print("    sudo dpkg -i intel-driver-compiler-npu_*.deb")
                    print("    sudo dpkg -i intel-fw-npu_*.deb")
                    print("    sudo dpkg -i intel-level-zero-npu_*.deb")
                    print()
                    print("  After installing, reboot and re-run this script.")
                if not args.cpu_only:
                    print("\n  Tip: run 'python install.py --cpu-only' to use CPU for now.")
        except ImportError:
            print("  OpenVINO import failed — check installation above.")
    else:
        print("\n  Step 3: NPU check skipped (--no-npu-check)")

    # ── Step 4: Update config if cpu-only ────────────────────────────────────
    if args.cpu_only:
        section("Step 4: Setting device to CPU (--cpu-only)")
        cfg = Path("config/settings.py")
        text = cfg.read_text()
        text = text.replace('INFERENCE_DEVICE = "NPU"', 'INFERENCE_DEVICE = "CPU"')
        cfg.write_text(text)
        print("  config/settings.py → INFERENCE_DEVICE = 'CPU'")
    else:
        section("Step 4: Device configuration")
        print("  INFERENCE_DEVICE = NPU  (default — uses Intel AI Boost)")
        print("  Edit config/settings.py to change if needed.")

    # ── Step 5: Create dirs ───────────────────────────────────────────────────
    section("Step 5: Creating project directories")
    for d in ("models", ".cache", "logs"):
        Path(d).mkdir(exist_ok=True)
        print(f"  {d}/  ✓")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"""
{'═'*60}
  Installation complete!

  Next steps:
    1. Download the AI model for your NPU:
         python opac.py --setup

    2. Start OPAC:
         python opac.py

    3. Run tests (no NPU required):
         python tests/test_phase1_2.py
{'═'*60}
""")


if __name__ == "__main__":
    main()
