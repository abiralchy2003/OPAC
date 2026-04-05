"""
OPAC Model Setup  (Phase 1)
============================
Downloads an NPU-optimised INT4 model from Hugging Face and
verifies that OpenVINO can load it on the NPU.

Run once:
    python opac.py --setup
Or directly:
    python core/model_setup.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from config.settings import DEFAULT_MODEL_DIR, DEFAULT_MODEL_REPO, INFERENCE_DEVICE
from utils.logger import get_logger

logger = get_logger("opac.setup")


def run_setup(
    model_repo: str  = DEFAULT_MODEL_REPO,
    model_dir:  Path = DEFAULT_MODEL_DIR,
    device:     str  = INFERENCE_DEVICE,
) -> bool:
    """
    1. Check OpenVINO is installed.
    2. Check NPU is visible.
    3. Download model from Hugging Face (already INT4 + OpenVINO IR format).
    4. Do a quick smoke-test generation.
    Returns True on success.
    """
    print("\n" + "═" * 60)
    print(" OPAC Setup — Phase 1: NPU Model Download & Verification")
    print("═" * 60)

    # ── Step 1: Check OpenVINO ────────────────────────────────────
    print("\n[1/4] Checking OpenVINO installation …")
    try:
        import openvino as ov
        print(f"      OpenVINO {ov.__version__}  ✓")
    except ImportError:
        print("      ✗ OpenVINO not found.")
        print("      Install: pip install openvino openvino-genai huggingface_hub")
        return False

    try:
        import openvino_genai  # noqa
        print("      OpenVINO GenAI  ✓")
    except ImportError:
        print("      ✗ openvino_genai not found.")
        print("      Install: pip install openvino-genai")
        return False

    # ── Step 2: Check NPU ─────────────────────────────────────────
    print(f"\n[2/4] Checking for {device} …")
    core    = ov.Core()
    devices = core.available_devices
    print(f"      Available devices: {', '.join(devices)}")

    if device not in devices:
        print(f"      ✗ {device} not found.")
        if device == "NPU":
            print("      Make sure the Intel NPU driver is installed:")
            print("      Windows: https://www.intel.com/content/www/us/en/download/794734/")
            print("      Linux:   https://github.com/intel/linux-npu-driver/releases")
        print(f"      Tip: set INFERENCE_DEVICE = 'CPU' in config/settings.py to test without NPU.")
        return False
    print(f"      {device}  ✓  (Intel AI Boost detected)")

    # ── Step 3: Download model ────────────────────────────────────
    print(f"\n[3/4] Downloading model: {model_repo}")
    print(f"      Target directory : {model_dir}")

    if model_dir.exists() and any(model_dir.iterdir()):
        print("      Directory already exists — skipping download.")
        print("      (Delete the directory and re-run --setup to force re-download.)")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("      ✗ huggingface_hub not installed.")
            print("      Install: pip install huggingface_hub")
            return False

        model_dir.mkdir(parents=True, exist_ok=True)
        print("      Downloading … (this may take several minutes)")
        t0 = time.time()
        snapshot_download(
            repo_id=model_repo,
            local_dir=str(model_dir),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        elapsed = time.time() - t0
        print(f"      Download complete in {elapsed/60:.1f} min  ✓")

    # ── Step 4: Smoke test ────────────────────────────────────────
    print(f"\n[4/4] Loading model on {device} and running smoke test …")
    print("      (First compilation takes ~60 s — result is cached for future runs)")
    try:
        import openvino_genai as ov_genai
        from config.settings import NPU_CONFIG

        cfg = NPU_CONFIG if device == "NPU" else {}
        t0  = time.time()
        pipe = ov_genai.LLMPipeline(str(model_dir), device, **cfg)
        elapsed = time.time() - t0
        print(f"      Model loaded in {elapsed:.1f}s  ✓")

        print("      Running test generation ('Hello, I am OPAC.') …")
        t0     = time.time()
        result = pipe.generate(
            "<|system|>\nYou are OPAC.<|end|>\n<|user|>\nSay hello in one sentence.<|end|>\n<|assistant|>\n",
            max_new_tokens=40,
        )
        elapsed = time.time() - t0
        print(f"      Response: {result.strip()}")
        tok = len(result.split())
        print(f"      ~{tok} tokens in {elapsed:.1f}s ({tok/max(elapsed,0.1):.1f} tok/s)  ✓")

    except Exception as exc:
        print(f"      ✗ Smoke test failed: {exc}")
        return False

    print("\n" + "═" * 60)
    print(" Setup complete!  OPAC is ready.")
    print(" Run:  python opac.py")
    print("═" * 60 + "\n")
    return True


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
