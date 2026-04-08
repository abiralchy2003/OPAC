"""
OPAC Summarizer  (Phase 2)
===========================
Reads documents, chunks long content, and calls the NPU engine.
Each chunk is sent as a fresh independent call — no shared session state.

Hierarchical combine: if there are many sections, summaries are first
grouped into batches, then each batch is combined, then the batch
results are combined into a final summary. This prevents the combine
prompt from exceeding the NPU context window (MAX_PROMPT_LEN).
"""

from __future__ import annotations

import time
from typing import List

from documents.loader import DocumentLoader
from utils.chunker import chunk_text
from utils.logger import get_logger
from config.settings import CHUNK_MAX_CHARS

logger = get_logger("opac.summarizer")

# How many section summaries to combine in one NPU call.
# 18 summaries of ~80 words each = ~1440 words ≈ 1800 tokens → overflows 2048.
# Batch of 5 summaries × ~80 words = ~400 words ≈ 500 tokens → safe for any MAX_PROMPT_LEN.
COMBINE_BATCH_SIZE = 5


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

        # ── Reduce: hierarchical combine ─────────────────────────────────────
        return self._hierarchical_combine(section_summaries, stream=stream)

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

    # ── hierarchical combine ─────────────────────────────────────────────────

    def _hierarchical_combine(self, summaries: List[str], stream: bool) -> str:
        """
        Combine summaries in batches to stay within the NPU context window.

        Example with 18 summaries and COMBINE_BATCH_SIZE=5:
          Round 1: [s1-s5] → b1, [s6-s10] → b2, [s11-s15] → b3, [s16-s18] → b4
          Round 2: [b1-b4] → final  (4 items, fits easily)

        If there is only one batch, it is combined directly (no extra round).
        """
        current = summaries
        round_num = 1

        while len(current) > 1:
            if len(current) <= COMBINE_BATCH_SIZE:
                # Final combine — show streaming output
                print(f"  [OPAC] Final combine ({len(current)} summaries) ...")
                prompt = self.engine.build_combine_prompt(current)
                return self._call(prompt, stream=stream)

            # Batch combine — silent (no streaming for intermediate steps)
            batches = _split_into_batches(current, COMBINE_BATCH_SIZE)
            print(f"  [OPAC] Combine round {round_num}: "
                  f"{len(current)} summaries → {len(batches)} batches ...")

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

        # Only one summary left (edge case: started with 1)
        return current[0] if current else "No content to summarise."

    # ── internal ─────────────────────────────────────────────────────────────

    def _call(self, user_message: str, stream: bool) -> str:
        """Send one user message to the engine as a fresh, stateless call."""
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


# ── utility ──────────────────────────────────────────────────────────────────

def _split_into_batches(items: List[str], batch_size: int) -> List[List[str]]:
    """Split a list into sub-lists of at most batch_size items."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
