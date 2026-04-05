"""
Direct NPU test — run this first to verify the model works independently.
Usage:  python test_npu_direct.py

This script bypasses all OPAC layers and talks to OpenVINO directly.
If this works, OPAC will work. If this fails, there is an NPU driver issue.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import DEFAULT_MODEL_DIR, NPU_CONFIG

print("=" * 55)
print("OPAC — Direct NPU Test")
print("=" * 55)
print(f"Model: {DEFAULT_MODEL_DIR}")
print()

# 1. Load
try:
    import openvino_genai as ov_genai
except ImportError:
    print("ERROR: openvino_genai not installed.")
    sys.exit(1)

if not DEFAULT_MODEL_DIR.exists():
    print("ERROR: Model not found. Run: python opac.py --setup")
    sys.exit(1)

print("Loading model on NPU ...")
t0   = time.time()
pipe = ov_genai.LLMPipeline(str(DEFAULT_MODEL_DIR), "NPU", **NPU_CONFIG)
print(f"Loaded in {time.time()-t0:.1f}s")

# 2. Build a minimal Qwen3 ChatML prompt — NO pre-filled think block
PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Say hello in exactly one sentence. /no_think<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print(f"\nPrompt sent:\n{PROMPT}")
print("-" * 55)
print("Response:")

cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = 60
cfg.do_sample      = False

collected = []
def _cb(tok):
    collected.append(tok)
    print(tok, end="", flush=True)
    return False

t0 = time.time()
pipe.generate(PROMPT, cfg, _cb)
elapsed = time.time() - t0
result  = "".join(collected).strip()

print(f"\n\nTokens: {len(result.split())}  Time: {elapsed:.1f}s")
print("=" * 55)

if result and len(result.split()) > 1 and "acleacle" not in result and "TokenName" not in result:
    print("RESULT: PASS — NPU is generating correct text")
else:
    print("RESULT: FAIL — garbage output detected")
    print("Try switching model in config/settings.py or update NPU driver")
