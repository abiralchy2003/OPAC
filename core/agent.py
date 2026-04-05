"""
OPAC Agent
==========
Central orchestrator. Owns the NPU engine and all sub-systems.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

from config.settings import DEFAULT_MODEL_DIR, INFERENCE_DEVICE
from core.npu_engine import NPUEngine
from core.summarizer import Summarizer
from utils.logger import get_logger

logger = get_logger("opac.agent")

_INTENT_HELP = re.compile(r"^(help|\?|commands)$", re.I)
_INTENT_QUIT = re.compile(r"^(quit|exit|bye|q)$",  re.I)
_INTENT_INFO = re.compile(r"^(info|status|device)$", re.I)
_INTENT_OPEN = re.compile(r"^(?:open|launch|start|run)\s+(.+)$", re.I)
_INTENT_SUM  = re.compile(
    r"^(?:summarize|summarise|summary of|read|explain)\s+(.+)$", re.I
)


class OPACAgent:
    def __init__(
        self,
        model_dir:       Path = DEFAULT_MODEL_DIR,
        device:          str  = INFERENCE_DEVICE,
        model_override:  str  = None,
        device_override: str  = None,
    ):
        if device_override:
            device = device_override.upper()
        if model_override:
            model_dir = DEFAULT_MODEL_DIR.parent / model_override

        self.engine     = NPUEngine(model_dir=model_dir, device=device)
        self.summarizer = Summarizer(self.engine)
        self._history: List[Dict] = []   # [{"role": "user"|"assistant", "content": "..."}]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def setup(self):
        from core.model_setup import run_setup
        run_setup()

    def is_ready(self) -> bool:
        return DEFAULT_MODEL_DIR.exists() and any(DEFAULT_MODEL_DIR.iterdir())

    def start(self):
        if not self.engine.loaded:
            self.engine.load()

    def stop(self):
        self.engine.unload()

    # ── one-shot ──────────────────────────────────────────────────────────────

    def summarize_file(self, path: str) -> str:
        self.start()
        return self.summarizer.summarize_file(path)

    def summarize_url(self, url: str) -> str:
        self.start()
        return self.summarizer.summarize_url(url)

    def chat(self, query: str) -> str:
        self.start()
        # Keep last 3 turns (6 messages) for context
        recent = self._history[-6:] if self._history else None
        response = self.engine.chat_turn(
            user_message=query,
            history=recent,
            streamer_callback=lambda tok: print(tok, end="", flush=True),
        )
        print()
        self._history.append({"role": "user",      "content": query})
        self._history.append({"role": "assistant",  "content": response})
        if len(self._history) > 12:
            self._history = self._history[-12:]
        return response

    # ── interactive REPL ──────────────────────────────────────────────────────

    def run_interactive(self):
        print("\n  Type a command, paste a file path, or paste a URL.")
        print("  Type 'help' for command list.  Type 'quit' to exit.\n")

        self.start()
        print(f"  [OPAC] Ready on {self.engine.device}  ✓\n")

        while True:
            try:
                user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  [OPAC] Goodbye.\n")
                break

            if not user_input:
                continue

            if _INTENT_QUIT.match(user_input):
                print("\n  [OPAC] Goodbye.\n")
                break

            if _INTENT_HELP.match(user_input):
                self._print_help()
                continue

            if _INTENT_INFO.match(user_input):
                self._print_status()
                continue

            self._handle_input(user_input)

        self.stop()

    # ── routing ───────────────────────────────────────────────────────────────

    def _handle_input(self, user_input: str):
        # Bare URL
        url_match = re.search(r"https?://\S+", user_input)
        if url_match:
            url = url_match.group(0)
            print(f"\n  [OPAC] Fetching: {url}")
            result = self.summarizer.summarize_url(url, stream=True)
            if result:
                print(f"\n  OPAC: {result}\n")
            return

        # Bare file path (with backslash, forward slash, or drive letter)
        clean = user_input.strip().strip('"').strip("'")
        if _looks_like_path(clean):
            print(f"\n  [OPAC] Summarising file: {clean}")
            result = self.summarizer.summarize_file(clean, stream=True)
            if result:
                print(f"\n  OPAC: {result}\n")
            return

        # "summarize <target>"
        m = _INTENT_SUM.match(user_input)
        if m:
            target = m.group(1).strip().strip('"').strip("'")
            url_in = re.match(r"https?://\S+", target)
            if url_in:
                result = self.summarizer.summarize_url(target, stream=True)
            elif _looks_like_path(target):
                result = self.summarizer.summarize_file(target, stream=True)
            else:
                result = f"Not found: {target}"
            if result:
                print(f"\n  OPAC: {result}\n")
            return

        # "open <app>" — Phase 5
        m = _INTENT_OPEN.match(user_input)
        if m:
            print(f"\n  OPAC: App launcher coming in Phase 5. (You asked to open: '{m.group(1)}')\n")
            return

        # General chat
        print("\n  OPAC: ", end="", flush=True)
        self.chat(user_input)
        print()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _print_help(self):
        print("""
  ┌─────────────────────────────────────────────────────────┐
  │  OPAC Commands                                          │
  ├─────────────────────────────────────────────────────────┤
  │  <file path>              Summarise a file              │
  │  <URL>                    Summarise a web page          │
  │  summarize <file/URL>     Same as above                 │
  │  open <app>               Open an application (Ph.5)   │
  │  <any question>           Ask OPAC anything             │
  │  info                     Show device/model status      │
  │  help                     Show this menu                │
  │  quit / exit              Exit OPAC                     │
  └─────────────────────────────────────────────────────────┘
""")

    def _print_status(self):
        status = "loaded  ✓" if self.engine.loaded else "not loaded"
        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  OPAC Status                                            │
  ├─────────────────────────────────────────────────────────┤
  │  Device   : {self.engine.device:<44} │
  │  Model    : {str(self.engine.model_dir.name):<44} │
  │  Pipeline : {status:<44} │
  └─────────────────────────────────────────────────────────┘
""")


def _looks_like_path(text: str) -> bool:
    """Return True if the string looks like a local file path."""
    import os
    from pathlib import Path as P
    # Windows drive letter or UNC
    if re.match(r"^[A-Za-z]:[/\\]", text):
        return True
    # Absolute Unix path
    if text.startswith("/"):
        return True
    # Relative path that exists on disk
    if P(text).exists():
        return True
    # Has a file extension and no spaces (likely a filename)
    if "." in text and " " not in text and len(text) < 260:
        return True
    return False
