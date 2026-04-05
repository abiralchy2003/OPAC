"""Print system and NPU availability information."""
import platform
import subprocess
import sys


def print_system_info():
    print("=" * 60)
    print("OPAC — System Information")
    print("=" * 60)
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"Machine     : {platform.machine()}")
    print(f"Python      : {sys.version.split()[0]}")
    print()

    # OpenVINO
    try:
        import openvino as ov
        print(f"OpenVINO    : {ov.__version__}  ✓")
        core = ov.Core()
        devices = core.available_devices
        print(f"OV Devices  : {', '.join(devices)}")
        npu_found = any("NPU" in d for d in devices)
        print(f"NPU present : {'YES  ✓ — AI Boost detected' if npu_found else 'NO   ✗ — NPU not found'}")
    except ImportError:
        print("OpenVINO    : NOT INSTALLED")
        print("             → Run: pip install openvino openvino-genai")
        print("             → Download NPU driver from https://github.com/intel/linux-npu-driver")

    # OpenVINO GenAI
    try:
        import openvino_genai as ov_genai  # noqa
        print("OV GenAI    : installed  ✓")
    except ImportError:
        print("OV GenAI    : NOT INSTALLED  → pip install openvino-genai")

    print()
    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"RAM Total   : {ram.total / 1e9:.1f} GB")
        print(f"RAM Free    : {ram.available / 1e9:.1f} GB")
    except ImportError:
        pass

    print("=" * 60)
