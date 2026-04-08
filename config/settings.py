"""
OPAC Configuration
==================
Central configuration. Edit this file to change models, device, and behaviour.
All paths are relative to the project root unless absolute.
"""

import os
import platform
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR  = ROOT_DIR / ".cache"
LOGS_DIR   = ROOT_DIR / "logs"

for d in (MODELS_DIR, CACHE_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── NPU / Inference device ───────────────────────────────────────────────────
# "NPU"  → Intel AI Boost (recommended — leaves CPU/GPU free)
# "GPU"  → Intel Arc (faster tokens, uses GPU)
# "CPU"  → fallback (slow, not recommended)
INFERENCE_DEVICE = "NPU"

# ── Model selection ──────────────────────────────────────────────────────────
# Recommended for 16 GB RAM:
#   "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"      ~2.5 GB RAM, ~12 tok/s on NPU
#   "OpenVINO/Qwen3-4B-int4-cw-ov"                 ~3.0 GB RAM, ~10 tok/s on NPU
#   "OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov" ~6.0 GB RAM, ~8  tok/s on NPU
DEFAULT_MODEL_REPO = "OpenVINO/Qwen3-8B-int4-cw-ov"
DEFAULT_MODEL_DIR  = MODELS_DIR / "qwen3-8b-int4-cw-npu"

# ── NPU generation config ────────────────────────────────────────────────────
NPU_CONFIG = {
    "MAX_PROMPT_LEN":   4096,
    "MIN_RESPONSE_LEN": 32,
}
MAX_NEW_TOKENS   = 512          # max tokens the model generates per response
TEMPERATURE      = 0.1          # low = focused/factual, high = creative
DO_SAMPLE        = False        # False = greedy (stable on NPU), True = sampling

# ── Chunking (for long documents) ───────────────────────────────────────────
# Documents longer than this are split into chunks and summarised in parts
CHUNK_MAX_CHARS  = 3000         # characters per chunk sent to LLM
CHUNK_OVERLAP    = 200          # overlap between chunks to preserve context

# ── Document reading ─────────────────────────────────────────────────────────
# Maximum characters extracted from a single document before chunking kicks in
DOC_MAX_CHARS    = 50_000

# ── Web scraping ─────────────────────────────────────────────────────────────
WEB_TIMEOUT_SEC  = 15
WEB_MAX_CHARS    = 8_000        # max chars extracted from a webpage

# ── Platform detection ───────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are OPAC, a concise and accurate local AI assistant. "
    "You summarize documents, answer questions, and help the user with their tasks. "
    "Keep responses clear and to the point. Do not make up information."
)

SUMMARIZE_PROMPT_TEMPLATE = (
    "Please provide a clear, concise summary of the following content. "
    "Highlight the main points, key findings, and important details. "
    "Keep the summary informative but brief.\n\n"
    "Content:\n{content}\n\n"
    "Summary:"
)

CHUNK_SUMMARY_PROMPT = (
    "Summarize this section of a larger document. Be concise.\n\n"
    "Section:\n{content}\n\nSection summary:"
)

COMBINE_SUMMARIES_PROMPT = (
    "Below are summaries of individual sections of a document. "
    "Combine them into one cohesive final summary.\n\n"
    "{summaries}\n\nFinal summary:"
)

CHAT_PROMPT_TEMPLATE = (
    "{history}"
    "User: {query}\n"
    "OPAC:"
)
