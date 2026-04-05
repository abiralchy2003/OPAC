"""
OPAC NPU Engine  (Phase 1 — Core)
==================================
Definitive implementation for Qwen3-4B-int4 on OpenVINO 2026 + Intel NPU.

Key decisions:
  - NO start_chat() / finish_chat() — stateful chat API corrupts after first call
  - Manual ChatML prompt building — the only reliable method on OV 2026 NPU
  - NO pre-filled <think> block — caused TokenNameIdentifier loop
  - /no_think appended to user message — Qwen3's official way to disable thinking
  - <think>...</think> stripped from output as safety net
  - DecodedResults unwrapped for OV 2026 compatibility
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, Callable, List, Dict

from config.settings import (
    DEFAULT_MODEL_DIR,
    INFERENCE_DEVICE,
    MAX_NEW_TOKENS,
    NPU_CONFIG,
    DO_SAMPLE,
    SYSTEM_PROMPT,
)
from utils.logger import get_logger

logger = get_logger("opac.npu_engine")

_THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$",          re.DOTALL)
_GARBAGE_RE    = re.compile(r"\b(TokenName\w+|acleacle\w*|aphore\w*)\b")


def _clean(text: str) -> str:
    """Strip Qwen3 thinking blocks, garbage tokens, repeated junk."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    text = _GARBAGE_RE.sub("", text)
    # Detect degenerate repetition — if same 4+ char sequence repeats 10+ times, truncate
    m = re.search(r"(.{4,}?)\1{10,}", text)
    if m:
        text = text[:m.start()].strip()
        logger.warning("Degenerate repetition detected and truncated.")
    text = text.replace("`", "")
    text = re.sub(r"[ \t]{3,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unwrap(result) -> str:
    """Handle OV 2025 (str) and OV 2026 (DecodedResults) return types."""
    if isinstance(result, str):
        return result
    # OV 2026: DecodedResults object
    for attr in ("texts", "strings", "values"):
        val = getattr(result, attr, None)
        if val:
            return val[0]
    return str(result)


def _build_chatml(
    user_message: str,
    system: str = SYSTEM_PROMPT,
    history: Optional[List[Dict]] = None,
) -> str:
    """
    Build a correct Qwen3 ChatML prompt.

    Format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {message} /no_think<|im_end|>
        <|im_start|>assistant
        {response}<|im_end|>     <- only for history turns
        ...
        <|im_start|>user
        {current message} /no_think<|im_end|>
        <|im_start|>assistant
                                 <- model fills from here

    Rules:
    - /no_think on every user turn suppresses chain-of-thought
    - NO pre-filled assistant content (caused TokenNameIdentifier bug)
    - History is interleaved user/assistant turns
    """
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]

    for msg in (history or []):
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|im_start|>user\n{content} /no_think<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

    # Current user turn — model generates from after this
    parts.append(f"<|im_start|>user\n{user_message} /no_think<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


class NPUEngine:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: str     = INFERENCE_DEVICE,
    ):
        self.model_dir = Path(model_dir)
        self.device    = device.upper()
        self._pipeline = None
        self._loaded   = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_dir}\n"
                "Run: python opac.py --setup"
            )
        logger.info(f"Loading model from {self.model_dir} onto {self.device} ...")
        logger.info("First compile ~30-60 s — cached after first run.")
        t0 = time.time()
        try:
            import openvino_genai as ov_genai
        except ImportError:
            raise ImportError("pip install openvino openvino-genai")

        device_cfg = NPU_CONFIG if self.device == "NPU" else {}
        self._pipeline = ov_genai.LLMPipeline(
            str(self.model_dir), self.device, **device_cfg
        )
        self._loaded = True
        logger.info(f"Model loaded on {self.device} in {time.time()-t0:.1f}s")

    def unload(self) -> None:
        self._pipeline = None
        self._loaded   = False
        logger.info("Model unloaded.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── generation ───────────────────────────────────────────────────────────

    def _make_gen_config(self, max_new_tokens: int):
        import openvino_genai as ov_genai
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.do_sample      = DO_SAMPLE
        return cfg

    def _run(
        self,
        prompt: str,
        max_new_tokens: int,
        streamer_callback: Optional[Callable[[str], None]],
    ) -> str:
        """Raw generation — no chat session, just the prompt string."""
        cfg = self._make_gen_config(max_new_tokens)

        if streamer_callback is not None:
            collected: list[str] = []
            in_think = [False]

            def _streamer(token: str) -> bool:
                collected.append(token)
                so_far = "".join(collected)
                if "<think>" in so_far and not in_think[0]:
                    in_think[0] = True
                if in_think[0]:
                    if "</think>" in so_far:
                        in_think[0] = False
                    return False
                streamer_callback(token)
                return False

            self._pipeline.generate(prompt, cfg, _streamer)
            return _clean("".join(collected))
        else:
            raw = self._pipeline.generate(prompt, cfg)
            return _clean(_unwrap(raw))

    # ── public interface ─────────────────────────────────────────────────────

    def _generate_chat(
        self,
        user_message: str,
        system: str                        = SYSTEM_PROMPT,
        max_new_tokens: int                = MAX_NEW_TOKENS,
        streamer_callback: Optional[Callable] = None,
        history: Optional[List[Dict]]      = None,
    ) -> str:
        """Main generation entry point used by Summarizer and Agent."""
        if not self._loaded:
            raise RuntimeError("Call engine.load() first.")

        prompt = _build_chatml(user_message, system=system, history=history)
        t0     = time.time()
        result = self._run(prompt, max_new_tokens, streamer_callback)

        elapsed   = time.time() - t0
        tok_count = len(result.split())
        logger.info(
            f"Generated ~{tok_count} tokens in {elapsed:.1f}s "
            f"(~{tok_count/max(elapsed,0.1):.1f} tok/s) on {self.device}"
        )
        return result

    def chat_turn(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        streamer_callback: Optional[Callable] = None,
    ) -> str:
        """Multi-turn conversation used by agent.py."""
        return self._generate_chat(
            user_message      = user_message,
            history           = history,
            streamer_callback = streamer_callback,
        )

    # legacy generate() — kept for backward compat with tests
    def generate(
        self,
        prompt: str,
        max_new_tokens: int                = MAX_NEW_TOKENS,
        streamer_callback: Optional[Callable] = None,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("Call engine.load() first.")
        return self._run(prompt, max_new_tokens, streamer_callback)

    # ── prompt builders used by Summarizer ───────────────────────────────────

    def build_summarize_prompt(self, content: str) -> str:
        from config.settings import SUMMARIZE_PROMPT_TEMPLATE
        return SUMMARIZE_PROMPT_TEMPLATE.format(content=content)

    def build_chunk_prompt(self, content: str) -> str:
        from config.settings import CHUNK_SUMMARY_PROMPT
        return CHUNK_SUMMARY_PROMPT.format(content=content)

    def build_combine_prompt(self, summaries: List[str]) -> str:
        from config.settings import COMBINE_SUMMARIES_PROMPT
        joined = "\n\n---\n\n".join(
            f"Section {i+1}:\n{s}" for i, s in enumerate(summaries)
        )
        return COMBINE_SUMMARIES_PROMPT.format(summaries=joined)

    # legacy — used by agent.chat() fallback
    def build_prompt(self, user_message: str, history: str = "") -> str:
        return user_message
