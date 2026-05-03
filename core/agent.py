"""
OPAC Agent  (Phase 1 + 2 + 3 + 3.5 + 4 + 5)
==============================================
Phase 4: Browser tab summarisation
Phase 5: App launcher
"""

from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path
from typing import List, Dict

from config.settings import (
    DEFAULT_MODEL_DIR, INFERENCE_DEVICE,
    VOICE_ENABLED, WAKE_WORD,
    WIKI_ENABLED, WIKI_CONTEXT_PROMPT,
    BROWSER_SUMMARIZE_PROMPT, BROWSER_MAX_CHARS,
    SYSTEM_PROMPT,
)
from core.npu_engine import NPUEngine
from core.summarizer import Summarizer
from utils.logger import get_logger

logger = get_logger("opac.agent")

# ── Intent patterns ────────────────────────────────────────────────────────────
_INTENT_HELP    = re.compile(r"^(help|\?|commands)$", re.I)
_INTENT_QUIT    = re.compile(r"^(quit|exit|bye|goodbye|q)$", re.I)
_INTENT_INFO    = re.compile(r"^(info|status|device|version)$", re.I)
_INTENT_CLEAR   = re.compile(r"^(clear|reset|forget|new chat)$", re.I)
_INTENT_TONE    = re.compile(r"^(be|talk|speak)\s+(casual|formal|friendly|professional).*$", re.I)
_INTENT_VOICE   = re.compile(r"^voice\s*(on|off)$", re.I)
_INTENT_WIKI    = re.compile(r"^(?:search|look up|find|wiki|wikipedia)\s+(.+)$", re.I)
_INTENT_SUM     = re.compile(r"^(?:summarize|summarise|summary of|read|explain)\s+(.+)$", re.I)

# Phase 4 — browser tab
_INTENT_TAB     = re.compile(
    r"^(?:summarize|summarise|read|what is|what'?s on|explain)\s+"
    r"(?:this\s+)?(?:tab|page|browser|current tab|current page|website|site|article)$",
    re.I
)
_INTENT_TAB2    = re.compile(
    r"^(?:tab|page|browser)\s+(?:summary|summarize|summarise)$", re.I
)

# Phase 5 — app launcher
_INTENT_OPEN    = re.compile(
    r"^(?:open|launch|start|run|load)\s+(.+)$", re.I
)
_INTENT_APPS    = re.compile(r"^(?:list apps?|show apps?|what apps?|available apps?).*$", re.I)

_CASUAL = re.compile(
    r"^(hi|hello|hey|how are you|what.?s up|good\s+(morning|afternoon|evening|night)|"
    r"howdy|sup|greetings|yo\b|morning|evening|afternoon|"
    r"how.?s (it going|everything)|what can you do|who are you|"
    r"tell me about yourself|introduce yourself|"
    r"thanks?|thank you|cheers|cool|nice|great|awesome|ok|okay|alright|"
    r"are you there|you there|can you hear me).*$",
    re.I
)

_TOPIC_STRIP = re.compile(
    r"^(?:can you |could you |please |do you know |tell me |"
    r"what is |what are |who is |who was |explain |describe |"
    r"give me info(?:rmation)? (?:on|about) |"
    r"tell me about |i want to know about |"
    r"i want to learn about |what do you know about )",
    re.I
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

        self._history: List[Dict] = []
        self._voice_active = False
        self._tone         = "auto"
        self._stt          = None
        self._tts          = None
        self._wakeword     = None
        self._voice_queue: queue.Queue = queue.Queue()

        # Phase 4 — browser
        self._browser = None

        # Phase 5 — launcher
        self._launcher = None

        # Phase 3.5 — Wikipedia
        self._wiki = None
        if WIKI_ENABLED:
            self._init_wiki()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def setup(self):
        from core.model_setup import run_setup
        run_setup()

    def is_ready(self) -> bool:
        return DEFAULT_MODEL_DIR.exists() and any(DEFAULT_MODEL_DIR.iterdir())

    def start(self):
        if not self.engine.loaded:
            self.engine.load()

    def stop(self):
        if self._wakeword:
            self._wakeword.stop()
        self.engine.unload()

    # ── Phase 3 — voice ────────────────────────────────────────────────────────

    def enable_voice(self) -> bool:
        try:
            from voice.stt import STTEngine
            from voice.tts import TTSEngine
            if not self._stt:
                self._stt = STTEngine()
                self._stt.load()
            if not self._tts:
                self._tts = TTSEngine()
                self._tts.load()
            self._voice_active = True
            logger.info(f"Voice enabled -- TTS: {self._tts.backend}")
            return True
        except Exception as e:
            logger.error(f"Voice init failed: {e}")
            print(f"\n  [OPAC] Voice not available: {e}\n")
            return False

    def enable_wake_word(self):
        if not self._stt:
            return
        try:
            from voice.wakeword import WakeWordDetector
            self._wakeword = WakeWordDetector(
                callback=self._on_voice_command,
                stt_engine=self._stt,
            )
            self._wakeword.start()
            print(f"  [OPAC] Wake word active -- say '{WAKE_WORD}' to activate\n")
        except Exception as e:
            logger.error(f"Wake word init failed: {e}")

    def _on_voice_command(self, text: str):
        """Background thread — only queues text, never processes it."""
        if text and text.strip():
            logger.info(f"Voice command queued: '{text}'")
            self._voice_queue.put(text.strip())

    def _drain_voice_queue(self):
        """Main thread — process all queued voice commands."""
        while not self._voice_queue.empty():
            try:
                text = self._voice_queue.get_nowait()
            except queue.Empty:
                break
            print(f"\n  You (voice): {text}\n", flush=True)
            if _INTENT_QUIT.match(text.strip()):
                print("\n  OPAC: Goodbye! Take care.\n")
                self._tts_speak("Goodbye! Take care.")
                return "quit"
            self._handle_input(text)
            print("  You: ", end="", flush=True)
        return None

    # ── Phase 3.5 — Wikipedia ──────────────────────────────────────────────────

    def _init_wiki(self):
        try:
            from voice.wiki import WikiEngine
            self._wiki = WikiEngine()
            self._wiki.setup()
        except Exception as e:
            logger.debug(f"Wikipedia init: {e}")

    # ── Phase 4 — Browser ──────────────────────────────────────────────────────

    def _init_browser(self):
        if self._browser is None:
            from actions.browser import BrowserEngine
            self._browser = BrowserEngine()

    def _summarize_current_tab(self) -> str:
        """Grab current browser tab and summarise it."""
        self._init_browser()
        print("\n  [OPAC] Grabbing current browser tab ...", flush=True)
        try:
            url, text = self._browser.get_current_tab()
        except RuntimeError as e:
            return str(e)
        except Exception as e:
            return f"Browser error: {e}"

        if not text.strip():
            return "The page appears to be empty or could not be read."

        # Truncate to limit
        if len(text) > BROWSER_MAX_CHARS:
            text = text[:BROWSER_MAX_CHARS]
            print(f"  [OPAC] Page truncated to {BROWSER_MAX_CHARS} chars", flush=True)

        print(f"  [OPAC] Page: {url}")
        print(f"  [OPAC] Content: {len(text)} chars — summarising ...\n")

        prompt = BROWSER_SUMMARIZE_PROMPT.format(url=url, content=text)
        print("  OPAC: ", end="", flush=True)

        collected = []
        def _cb(tok):
            collected.append(tok)
            print(tok, end="", flush=True)
            return False

        self.engine._generate_chat(
            user_message=prompt,
            streamer_callback=_cb,
        )
        print("\n")
        response = "".join(collected).strip()
        self._tts_speak(response)
        if response:
            self._history.append({"role": "user",      "content": f"Summarise {url}"})
            self._history.append({"role": "assistant",  "content": response})
        return response

    # ── Phase 5 — App Launcher ────────────────────────────────────────────────

    def _init_launcher(self):
        if self._launcher is None:
            from actions.launcher import AppLauncher
            self._launcher = AppLauncher()
            print("  [OPAC] App launcher ready\n", flush=True)

    def _open_app(self, app_name: str) -> str:
        """Find and launch an application."""
        self._init_launcher()
        print(f"\n  [OPAC] Looking for '{app_name}' ...", flush=True)
        success, message = self._launcher.open(app_name)
        self._tts_speak(message)
        return message

    def _list_apps(self) -> str:
        """List all launchable apps."""
        self._init_launcher()
        _, message = self._launcher.list_apps()
        return message

    # ── public API ─────────────────────────────────────────────────────────────

    def summarize_file(self, path: str) -> str:
        self.start()
        return self.summarizer.summarize_file(path)

    def summarize_url(self, url: str) -> str:
        self.start()
        return self.summarizer.summarize_url(url)

    def chat(self, query: str) -> str:
        self.start()
        tone_system    = self._build_tone_system(query)
        enriched_query = self._enrich_with_wiki(query)
        recent         = self._history[-6:] if self._history else None

        collected = []
        def _cb(tok):
            collected.append(tok)
            print(tok, end="", flush=True)
            return False

        self.engine._generate_chat(
            user_message=enriched_query,
            system=tone_system,
            history=recent,
            streamer_callback=_cb,
        )
        print()
        response = "".join(collected).strip()
        if response:
            self._history.append({"role": "user",      "content": query})
            self._history.append({"role": "assistant",  "content": response})
            if len(self._history) > 20:
                self._history = self._history[-20:]
            self._tts_speak(response)
        return response

    def _enrich_with_wiki(self, query: str) -> str:
        if not self._wiki or not self._wiki.available:
            return query
        if not self._wiki.is_factual_query(query):
            return query
        topic   = _extract_topic(query)
        results = self._wiki.search(topic)
        if not results:
            return query
        ctx = self._wiki.format_context(results)
        logger.info(f"Wikipedia: '{topic}' ({len(results)} results)")
        return WIKI_CONTEXT_PROMPT.format(wiki_context=ctx, question=query)

    # ── REPL ───────────────────────────────────────────────────────────────────

    def run_interactive(self, voice: bool = False):
        print("\n  Type anything to chat, paste a file path, or a URL.")
        print("  Type 'help' for all commands.  Type 'quit' to exit.\n")

        self.start()
        print(f"  [OPAC] Ready on {self.engine.device}  checkmark\n")

        if voice or VOICE_ENABLED:
            if self.enable_voice():
                print(f"  [OPAC] Voice enabled  (TTS: {self._tts.backend})")
                self.enable_wake_word()

        while True:
            result = self._drain_voice_queue()
            if result == "quit":
                break

            try:
                typed = self._input_with_voice_check("  You: ")
            except (EOFError, KeyboardInterrupt):
                print("\n\n  [OPAC] Goodbye!\n")
                break

            if typed is None:
                continue

            raw = typed.strip()
            if not raw:
                continue

            if _INTENT_QUIT.match(raw):
                print("\n  OPAC: Goodbye! Take care.\n")
                self._tts_speak("Goodbye! Take care.")
                break

            if _INTENT_HELP.match(raw):
                self._print_help()
                continue

            if _INTENT_INFO.match(raw):
                self._print_status()
                continue

            if _INTENT_CLEAR.match(raw):
                self._history.clear()
                print("  [OPAC] Conversation memory cleared.\n")
                continue

            m = _INTENT_TONE.match(raw)
            if m:
                self._tone = m.group(2).lower()
                print(f"  [OPAC] Tone set to: {self._tone}\n")
                continue

            m = _INTENT_VOICE.match(raw)
            if m:
                if m.group(1).lower() == "on":
                    if self.enable_voice():
                        print(f"  [OPAC] Voice on (TTS: {self._tts.backend})\n")
                        self.enable_wake_word()
                else:
                    self._voice_active = False
                    print("  [OPAC] Voice off\n")
                continue

            self._handle_input(raw)

        self.stop()

    def _input_with_voice_check(self, prompt: str):
        result_holder = [None]
        done_event    = threading.Event()

        def _read():
            try:
                result_holder[0] = input(prompt)
            except (EOFError, KeyboardInterrupt):
                result_holder[0] = "\x03"
            finally:
                done_event.set()

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        while not done_event.is_set():
            done_event.wait(timeout=0.2)
            if not self._voice_queue.empty():
                result = self._drain_voice_queue()
                if result == "quit":
                    raise EOFError
                return None

        raw = result_holder[0]
        if raw == "\x03":
            raise KeyboardInterrupt
        return raw

    def run_voice_mode(self):
        if not self.enable_voice():
            print("  [OPAC] Cannot start voice mode.")
            return
        self.start()
        self.enable_wake_word()
        print(f"\n  [OPAC] Voice mode active. Say '{WAKE_WORD}' to start.")
        print("  Press Ctrl+C to exit.\n")
        self._tts_speak("Voice mode active. Say hey opac to talk to me.")
        try:
            while True:
                result = self._drain_voice_queue()
                if result == "quit":
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n  [OPAC] Goodbye.\n")
        finally:
            self.stop()

    # ── router ─────────────────────────────────────────────────────────────────

    def _handle_input(self, user_input: str):
        # 1. Greetings
        if _CASUAL.match(user_input.strip()):
            print("\n  OPAC: ", end="", flush=True)
            self.chat(user_input)
            print()
            return

        # 2. Phase 4 — browser tab
        if _INTENT_TAB.match(user_input.strip()) or _INTENT_TAB2.match(user_input.strip()):
            self.start()
            self._summarize_current_tab()
            return

        # 3. Phase 5 — list apps
        if _INTENT_APPS.match(user_input.strip()):
            result = self._list_apps()
            print(result)
            return

        # 4. Phase 5 — open app
        m = _INTENT_OPEN.match(user_input)
        if m:
            app_name = m.group(1).strip()
            # Don't treat file paths as app names
            if not _looks_like_path(app_name) and not app_name.startswith("http"):
                self.start()
                result = self._open_app(app_name)
                print(f"\n  OPAC: {result}\n")
                return

        # 5. Wikipedia explicit search
        m = _INTENT_WIKI.match(user_input)
        if m:
            self._do_wiki_search(m.group(1).strip())
            return

        # 6. URL
        url_match = re.search(r"https?://\S+", user_input)
        if url_match:
            url = url_match.group(0)
            print(f"\n  [OPAC] Fetching: {url}")
            result = self.summarizer.summarize_url(url, stream=True)
            if result:
                print(f"\n  OPAC: {result}\n")
                self._tts_speak(result)
            return

        # 7. File path
        clean = user_input.strip().strip('"').strip("'")
        if _looks_like_path(clean):
            print(f"\n  [OPAC] Summarising: {clean}")
            result = self.summarizer.summarize_file(clean, stream=True)
            if result:
                print(f"\n  OPAC: {result}\n")
                self._tts_speak(result)
            return

        # 8. Summarize keyword
        m = _INTENT_SUM.match(user_input)
        if m:
            target = m.group(1).strip().strip('"').strip("'")
            if re.match(r"https?://\S+", target):
                result = self.summarizer.summarize_url(target, stream=True)
            elif _looks_like_path(target):
                result = self.summarizer.summarize_file(target, stream=True)
            else:
                result = f"Path or URL not found: {target}"
            if result:
                print(f"\n  OPAC: {result}\n")
                self._tts_speak(result)
            return

        # 9. General chat
        print("\n  OPAC: ", end="", flush=True)
        self.chat(user_input)
        print()

    # ── Wikipedia ──────────────────────────────────────────────────────────────

    def _do_wiki_search(self, query: str):
        if not self._wiki or not self._wiki.available:
            print("\n  [OPAC] Wikipedia not available. Run: pip install wikipedia-api\n")
            print("\n  OPAC: ", end="", flush=True)
            self.chat(f"Tell me about {query}")
            print()
            return
        topic   = _extract_topic(query)
        print(f"\n  [OPAC] Searching Wikipedia: '{topic}' ...")
        results = self._wiki.search(topic)
        if not results:
            print(f"  [OPAC] No results -- answering from knowledge ...\n")
            print("  OPAC: ", end="", flush=True)
            self.chat(f"Tell me about {topic}")
            print()
            return
        from config.settings import WIKI_CONTEXT_PROMPT
        ctx    = self._wiki.format_context(results)
        prompt = WIKI_CONTEXT_PROMPT.format(wiki_context=ctx,
                                            question=f"Tell me about {topic}")
        print(f"  [OPAC] Found {len(results)} article(s) ...\n")
        print("  OPAC: ", end="", flush=True)
        collected = []
        def _cb(tok):
            collected.append(tok)
            print(tok, end="", flush=True)
            return False
        self.engine._generate_chat(user_message=prompt, streamer_callback=_cb)
        print("\n")
        response = "".join(collected).strip()
        self._tts_speak(response)
        if response:
            self._history.append({"role": "user",      "content": query})
            self._history.append({"role": "assistant",  "content": response})

    # ── tone ───────────────────────────────────────────────────────────────────

    def _build_tone_system(self, msg: str) -> str:
        if self._tone == "casual":
            hint = "\n\nTone: Be casual, warm, and friendly."
        elif self._tone in ("formal", "professional"):
            hint = "\n\nTone: Be precise and professional."
        else:
            hint = self._auto_detect_tone(msg)
        return SYSTEM_PROMPT + hint

    def _auto_detect_tone(self, text: str) -> str:
        casual = len(re.findall(
            r"\b(hey|hi|lol|haha|btw|cool|awesome|wanna|gonna|kinda|ya|yep|nope|dunno)\b",
            text, re.I))
        formal = len(re.findall(
            r"\b(please|could you|would you|kindly|regarding|furthermore|therefore)\b",
            text, re.I))
        if casual > formal:
            return "\n\nTone: User is casual -- respond warmly, like a helpful friend."
        elif formal > casual:
            return "\n\nTone: User is formal -- respond precisely and professionally."
        return ""

    def _tts_speak(self, text: str) -> None:
        if self._voice_active and self._tts and self._tts.loaded:
            self._tts.speak(text)

    def _print_help(self):
        print("""
  +---------------------------------------------------------------+
  |  OPAC Commands                                                |
  +---------------------------------------------------------------+
  |  CONVERSATION                                                 |
  |    <anything>             Chat freely                         |
  |    be casual / be formal  Change tone                         |
  |    clear                  Clear memory                        |
  |                                                               |
  |  DOCUMENTS & WEB                                             |
  |    <file path>            Summarise a file                    |
  |    <URL>                  Summarise a web page                |
  |    summarize <file/URL>   Summarise with keyword              |
  |    search <topic>         Search Wikipedia                    |
  |                                                               |
  |  PHASE 4 - BROWSER                                           |
  |    summarize tab          Summarise current browser tab       |
  |    summarize page         Same as above                       |
  |    what is this page      Same as above                       |
  |                                                               |
  |  PHASE 5 - APP LAUNCHER                                      |
  |    open <app>             Open any installed app              |
  |    launch <app>           Same as above                       |
  |    list apps              Show all launchable apps            |
  |    Examples:                                                  |
  |      open chrome          open vs code                        |
  |      open spotify         open notepad                        |
  |      launch discord       start calculator                    |
  |                                                               |
  |  VOICE                                                        |
  |    voice on / voice off   Toggle voice output                 |
  |    Say "hey opac"         Activate voice input                |
  |                                                               |
  |  SYSTEM                                                       |
  |    info                   Show status                         |
  |    help                   Show this menu                      |
  |    quit / exit            Exit OPAC                           |
  +---------------------------------------------------------------+
""")

    def _print_status(self):
        wiki_ok  = "available" if (self._wiki and self._wiki.available) else "not available"
        wiki_mode = getattr(self._wiki, "_mode", "none") if self._wiki else "none"
        browser_ok = "ready" if self._browser else "not initialised (auto-loads on use)"
        launcher_ok = "ready" if self._launcher else "not initialised (auto-loads on use)"
        print(f"""
  Device    : {self.engine.device}
  Model     : {self.engine.model_dir.name}
  Pipeline  : {"loaded" if self.engine.loaded else "not loaded"}
  Voice     : {"on" if self._voice_active else "off"}
  Wikipedia : {wiki_ok} ({wiki_mode})
  Browser   : {browser_ok}
  Launcher  : {launcher_ok}
  Tone      : {self._tone}
  Memory    : {len(self._history)//2} turn(s)
""")


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_topic(query: str) -> str:
    clean = query.strip().rstrip("?").strip()
    clean = _TOPIC_STRIP.sub("", clean).strip()
    m = re.match(r"^(?:about|on|regarding)\s+(.+)$", clean, re.I)
    if m:
        clean = m.group(1).strip()
    return clean if clean else query


def _looks_like_path(text: str) -> bool:
    from pathlib import Path as P
    if re.match(r"^[A-Za-z]:[/\\]", text):
        return True
    if text.startswith("/"):
        return True
    if P(text).exists():
        return True
    if "." in text and " " not in text and len(text) < 260 and not text.endswith("?"):
        ext = text.rsplit(".", 1)[-1].lower()
        if ext in ("pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
                   "txt", "md", "csv", "html", "htm"):
            return True
    return False
