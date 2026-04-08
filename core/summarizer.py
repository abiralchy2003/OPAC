"""
OPAC Summarizer  (Phase 2)
===========================
Reads documents, chunks long content, and calls the NPU engine.
Each chunk is sent as a fresh independent call — no shared session state.
"""

from __future__ import annotations

import time
from typing import Optional

from documents.loader import DocumentLoader, DocumentResult
from utils.chunker import chunk_text
from utils.logger import get_logger
from config.settings import CHUNK_MAX_CHARS

logger = get_logger("opac.summarizer")


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

        # Multi-chunk: summarise each independently, then combine
        print(f"\n  [OPAC] Long document — summarising in {len(chunks)} sections ...")
        section_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  [OPAC] Processing section {i}/{len(chunks)} ...", end="", flush=True)
            summary = self._call(self.engine.build_chunk_prompt(chunk), stream=False)
            # Guard: if model produced nothing useful, use a placeholder
            if not summary or len(summary.split()) < 3:
                summary = f"(section {i}: content could not be extracted)"
            section_summaries.append(summary)
            print(f" done ({len(summary.split())} words)")

        print("  [OPAC] Combining section summaries ...")
        return self._call(
            self.engine.build_combine_prompt(section_summaries), stream=stream
        )

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

    # ── internal ─────────────────────────────────────────────────────────────

    def _call(self, user_message: str, stream: bool) -> str:
        """Send one user message to the engine as a fresh, independent call."""
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
