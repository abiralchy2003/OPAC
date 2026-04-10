"""
OPAC Configuration
==================
Central configuration. Edit this file to change models, device, and behaviour.
All paths are relative to the project root unless absolute.
"""

import platform
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR  = ROOT_DIR / ".cache"
LOGS_DIR   = ROOT_DIR / "logs"
DATA_DIR   = ROOT_DIR / "data"        # Phase 3.5 — Wikipedia dumps

for d in (MODELS_DIR, CACHE_DIR, LOGS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── NPU / Inference device ────────────────────────────────────────────────────
INFERENCE_DEVICE = "NPU"   # NPU | GPU | CPU

# ── Model selection ───────────────────────────────────────────────────────────
DEFAULT_MODEL_REPO = "OpenVINO/Qwen3-8B-int4-cw-ov"
DEFAULT_MODEL_DIR  = MODELS_DIR / "qwen3-8b-int4-cw-npu"

# ── NPU generation config ─────────────────────────────────────────────────────
NPU_CONFIG = {
    "MAX_PROMPT_LEN":   4096,
    "MIN_RESPONSE_LEN": 32,
}
MAX_NEW_TOKENS = 512
DO_SAMPLE      = False   # greedy — stable on NPU

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MAX_CHARS = 3000
CHUNK_OVERLAP   = 200
DOC_MAX_CHARS   = 50_000

# ── Web scraping ──────────────────────────────────────────────────────────────
WEB_TIMEOUT_SEC = 15
WEB_MAX_CHARS   = 8_000

# ── Platform ──────────────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# ── Voice (Phase 3) ───────────────────────────────────────────────────────────
VOICE_ENABLED       = False          # set True after installing voice libraries
WAKE_WORD           = "hey opac"     # spoken trigger phrase
WHISPER_MODEL_SIZE  = "base"         # tiny | base | small | medium
WHISPER_DEVICE      = "cpu"          # whisper runs on CPU (leaves NPU for LLM)
TTS_ENGINE          = "auto"         # auto | piper | sapi | espeak
PIPER_VOICE_MODEL   = ""             # path to .onnx piper voice model file
MIC_SILENCE_TIMEOUT = 3.0            # seconds of silence before STT stops
MIC_ENERGY_THRESHOLD = 300           # microphone sensitivity (lower = more sensitive)

# ── Wikipedia (Phase 3.5) ─────────────────────────────────────────────────────
WIKI_DB_PATH        = DATA_DIR / "wiki.db"   # SQLite DB built from Wikipedia dump
WIKI_MAX_RESULTS    = 3                       # articles to retrieve per query
WIKI_SNIPPET_CHARS  = 1500                    # chars per article fed to LLM
WIKI_ENABLED        = True                    # use Wikipedia search when relevant

# ── Conversation system prompt ────────────────────────────────────────────────
# This is the core personality of OPAC.
# It is injected into every chat prompt so the model knows how to behave.
SYSTEM_PROMPT = """You are OPAC (Objective-driven Planning and Action Coordinator), \
a smart, friendly, and capable local AI assistant running entirely on the user's \
Intel NPU — completely offline and private.

Personality and behaviour:
- Be natural and conversational. Match the user's tone: if they are casual and \
friendly, respond the same way. If they are formal or technical, be precise and \
professional.
- For greetings like "hello", "hi", "how are you" — respond warmly and naturally, \
like a helpful friend. Do NOT say you cannot answer simple questions.
- For casual chat — be relaxed, personable, and engaging. You can use light humour \
when appropriate.
- For technical or academic questions — be thorough, structured, and accurate.
- For document summaries — be concise and highlight what matters most.
- Always reason through your answers. If a question needs thought, think it through \
step by step before answering.
- Be honest. If you do not know something, say so clearly rather than guessing.
- Keep responses focused — do not pad with unnecessary filler words.
- You run fully offline on the user's Intel NPU. Mention this proudly if asked \
about your setup.
- Your creator is an IT student who built you as a personal AI project."""

# ── Task-specific prompts ─────────────────────────────────────────────────────
SUMMARIZE_PROMPT_TEMPLATE = (
    "Please provide a clear, concise summary of the following content. "
    "Highlight the main points, key findings, and important details. "
    "Keep the summary informative but brief.\n\n"
    "Content:\n{content}\n\nSummary:"
)

CHUNK_SUMMARY_PROMPT = (
    "Summarize this section of a larger document. Be concise.\n\n"
    "Section:\n{content}\n\nSection summary:"
)

COMBINE_SUMMARIES_PROMPT = (
    "Below are summaries of individual sections of a document. "
    "Combine them into one cohesive, well-structured final summary. "
    "Do not repeat information unnecessarily.\n\n"
    "{summaries}\n\nFinal summary:"
)

WIKI_CONTEXT_PROMPT = (
    "The following information from Wikipedia may be relevant to answer the question.\n\n"
    "{wiki_context}\n\n"
    "Using the above context and your own knowledge, answer this question:\n{question}"
)
