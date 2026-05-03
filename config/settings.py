"""OPAC Configuration — Phase 1-5"""

import platform
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR  = ROOT_DIR / ".cache"
LOGS_DIR   = ROOT_DIR / "logs"
DATA_DIR   = ROOT_DIR / "data"

for d in (MODELS_DIR, CACHE_DIR, LOGS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

INFERENCE_DEVICE   = "NPU"
DEFAULT_MODEL_REPO = "OpenVINO/Qwen3-8B-int4-cw-ov"
DEFAULT_MODEL_DIR  = MODELS_DIR / "qwen3-8b-int4-cw-npu"

NPU_CONFIG = {
    "MAX_PROMPT_LEN":   4096,
    "MIN_RESPONSE_LEN": 32,
}
MAX_NEW_TOKENS = 768
DO_SAMPLE      = False

CHUNK_MAX_CHARS = 3000
CHUNK_OVERLAP   = 200
DOC_MAX_CHARS   = 50_000
WEB_TIMEOUT_SEC = 15
WEB_MAX_CHARS   = 8_000

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# Phase 3 — Voice
VOICE_ENABLED        = False
WAKE_WORD            = "hey opac"
WHISPER_MODEL_SIZE   = "base"
WHISPER_DEVICE       = "cpu"
TTS_ENGINE           = "auto"
PIPER_VOICE_MODEL    = ""
MIC_SILENCE_TIMEOUT  = 3.0
MIC_ENERGY_THRESHOLD = 300

# Phase 3.5 — Wikipedia
WIKI_DB_PATH     = DATA_DIR / "wiki.db"
WIKI_MAX_RESULTS = 3
WIKI_SNIPPET_CHARS = 1500
WIKI_ENABLED     = True

# Phase 4 — Browser
BROWSER_DEBUG_PORT = 9222       # Chrome/Edge remote debugging port
BROWSER_MAX_CHARS  = 12_000    # max chars from page to send to LLM

# Phase 5 — App Launcher
LAUNCHER_FUZZY_THRESHOLD = 0.60  # how similar name must be to match

# System prompt
SYSTEM_PROMPT = """You are OPAC (Objective-driven Planning and Action Coordinator), \
a smart, friendly, and capable local AI assistant running entirely on the user's \
Intel NPU — completely offline and private.

Personality and behaviour:
- Be natural and conversational. Match the user's tone: casual or formal.
- For greetings — respond warmly and naturally, like a helpful friend.
- For technical questions — be thorough, structured, and accurate.
- Always reason through your answers before responding.
- Be honest. If you do not know something, say so clearly.
- Keep responses focused and avoid unnecessary filler.
- You run fully offline on the user's Intel NPU."""

SUMMARIZE_PROMPT_TEMPLATE = (
    "Please provide a clear, concise summary of the following content. "
    "Highlight the main points and important details.\n\n"
    "Content:\n{content}\n\nSummary:"
)
CHUNK_SUMMARY_PROMPT = (
    "Summarize this section concisely.\n\nSection:\n{content}\n\nSection summary:"
)
COMBINE_SUMMARIES_PROMPT = (
    "Combine these section summaries into one cohesive final summary.\n\n"
    "{summaries}\n\nFinal summary:"
)
WIKI_CONTEXT_PROMPT = (
    "The following Wikipedia information may help answer the question.\n\n"
    "{wiki_context}\n\n"
    "Using the above and your own knowledge, answer:\n{question}"
)
BROWSER_SUMMARIZE_PROMPT = (
    "Please summarise the following web page content. "
    "Include the main topic, key points, and any important details.\n\n"
    "Page URL: {url}\n\nContent:\n{content}\n\nSummary:"
)
