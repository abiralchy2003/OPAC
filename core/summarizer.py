"""
OPAC Summarizer  (Phase 2)
===========================
Smart combine strategy:
  - If all section summaries fit within the NPU context window → combine in one
    call (faster, single NPU pass).
  - If they overflow → hierarchical batching (groups of COMBINE_BATCH_SIZE).

The threshold is estimated from the actual text length of the summaries,
using a conservative 1.5 chars-per-token ratio so we never get close
to the hard MAX_PROMPT_LEN limit.
"""

from __future__ import annotations

import time
from typing import List

from documents.loader import DocumentLoader
from utils.chunker import chunk_text
from utils.logger import get_logger
from config.settings import CHUNK_MAX_CHARS, NPU_CONFIG

logger = get_logger("opac.summarizer")

# How many section summaries per batch when hierarchical mode is needed
COMBINE_BATCH_SIZE = 5

# Conservative estimate: chars per token (Qwen3 averages ~4, we use 3 to be safe)
CHARS_PER_TOKEN = 3

# Reserve this many tokens for the combine prompt template overhead + response
PROMPT_OVERHEAD_TOKENS = 300

def _max_prompt_len() -> int:
    """Read MAX_PROMPT_LEN from NPU config at runtime."""
    return int(NPU_CONFIG.get("MAX_PROMPT_LEN", 2048))

def _fits_in_one_call(summaries: List[str]) -> bool:
    """
    Return True if all summaries can be combined in a single NPU call
    without overflowing MAX_PROMPT_LEN.
    """
    total_chars  = sum(len(s) for s in summaries)
    est_tokens   = total_chars // CHARS_PER_TOKEN
    available    = _max_prompt_len() - PROMPT_OVERHEAD_TOKENS
    fits         = est_tokens <= available
    logger.info(
        f"Combine check: ~{est_tokens} tokens needed, "
        f"{available} available (MAX_PROMPT_LEN={_max_prompt_len()}) "
        f"→ {'single call' if fits else 'hierarchical batching'}"
    )
    return fits


class Summarizer:
    def __init__(self, engine):
        self.engine = engine
        self.loader = DocumentLoader()

    # ── public ───────────────────────────────────────────────────────────────

    def summarize_text(self, text: str, stream: bool = True) -> str:
        if not text.strip():
            return "Nothing to summarise — the document appears to be empty."

        chunks = chunk_text(text, max_chars=CHUNK_MAX_CHARS)
        logger.info(f"Summarising {len(text)} chars in {len(chunks)} chunk(s)")

        if len(chunks) == 1:
            return self._call(self.engine.build_summarize_prompt(chunks[0]), stream=stream)

        # ── Map: summarise each chunk independently ───────────────────────────
        print(f"\n  [OPAC] Long document — summarising in {len(chunks)} sections ...")
        section_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  [OPAC] Processing section {i}/{len(chunks)} ...", end="", flush=True)
            summary = self._call(self.engine.build_chunk_prompt(chunk), stream=False)
            if not summary or len(summary.split()) < 3:
                summary = f"(section {i}: content could not be extracted)"
            section_summaries.append(summary)
            print(f" done ({len(summary.split())} words)")

        # ── Reduce: smart combine ─────────────────────────────────────────────
        return self._smart_combine(section_summaries, stream=stream)

    def summarize_file(self, path: str, stream: bool = True) -> str:
        t0 = time.time()
        try:
            doc = self.loader.load(path)
        except Exception as exc:
            return f"Error reading file: {exc}"
        logger.info(str(doc))
        print(f"\n  [OPAC] Read: {doc}  ({time.time()-t0:.1f}s)")
        return self.summarize_text(doc.text, stream=stream)

    def summarize_url(self, url: str, stream: bool = True) -> str:
        t0 = time.time()
        try:
            doc = self.loader.load(url)
        except Exception as exc:
            return f"Error fetching URL: {exc}"
        logger.info(str(doc))
        print(f"\n  [OPAC] Fetched: {doc}  ({time.time()-t0:.1f}s)")
        return self.summarize_text(doc.text, stream=stream)

    # ── smart combine ─────────────────────────────────────────────────────────

    def _smart_combine(self, summaries: List[str], stream: bool) -> str:
        """
        Choose combine strategy based on estimated token count vs MAX_PROMPT_LEN.

        MAX_PROMPT_LEN = 2048  and summaries overflow  → hierarchical batching
        MAX_PROMPT_LEN = 4096  and summaries fit        → single call (old method)
        MAX_PROMPT_LEN = 4096  and summaries overflow   → hierarchical batching
        """
        if _fits_in_one_call(summaries):
            # ── Single call (old method) ──────────────────────────────────────
            print(f"  [OPAC] Combining {len(summaries)} summaries in one call ...")
            prompt = self.engine.build_combine_prompt(summaries)
            return self._call(prompt, stream=stream)
        else:
            # ── Hierarchical batching ─────────────────────────────────────────
            print(f"  [OPAC] {len(summaries)} summaries too large for one call "
                  f"(MAX_PROMPT_LEN={_max_prompt_len()}) — using batch combine ...")
            return self._hierarchical_combine(summaries, stream=stream)

    def _hierarchical_combine(self, summaries: List[str], stream: bool) -> str:
        """
        Repeatedly combine in batches until only one summary remains.

        Example with 18 summaries, batch size 5:
          Round 1: [1-5]→b1  [6-10]→b2  [11-15]→b3  [16-18]→b4
          Round 2: [b1-b4] fits in one call → final summary
        """
        current   = summaries
        round_num = 1

        while len(current) > 1:
            if _fits_in_one_call(current):
                print(f"  [OPAC] Final combine ({len(current)} summaries) ...")
                prompt = self.engine.build_combine_prompt(current)
                return self._call(prompt, stream=stream)

            batches = _split_into_batches(current, COMBINE_BATCH_SIZE)
            print(f"  [OPAC] Combine round {round_num}: "
                  f"{len(current)} summaries → {len(batches)} batches of {COMBINE_BATCH_SIZE} ...")

            combined = []
            for j, batch in enumerate(batches, 1):
                print(f"  [OPAC]   Batch {j}/{len(batches)} ...", end="", flush=True)
                prompt  = self.engine.build_combine_prompt(batch)
                result  = self._call(prompt, stream=False)
                if not result or len(result.split()) < 3:
                    result = f"(batch {j}: could not be combined)"
                combined.append(result)
                print(f" done ({len(result.split())} words)")

            current   = combined
            round_num += 1

        return current[0] if current else "No content to summarise."

    # ── internal ─────────────────────────────────────────────────────────────

    def _call(self, user_message: str, stream: bool) -> str:
        if stream:
            print("\n  OPAC: ", end="", flush=True)
            result = self.engine._generate_chat(
                user_message=user_message,
                streamer_callback=lambda tok: print(tok, end="", flush=True),
            )
            print()
            return result
        else:
            return self.engine._generate_chat(user_message=user_message)


# ── utility ───────────────────────────────────────────────────────────────────

def _split_into_batches(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]