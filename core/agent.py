"""
OPAC Agent  (Phase 1 + 2 + 3 + 3.5)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

from config.settings import (
    DEFAULT_MODEL_DIR, INFERENCE_DEVICE,
    VOICE_ENABLED, WAKE_WORD,
    WIKI_ENABLED, WIKI_CONTEXT_PROMPT,
    SYSTEM_PROMPT,
)
from core.npu_engine import NPUEngine
from core.summarizer import Summarizer
from utils.logger import get_logger

logger = get_logger("opac.agent")

_INTENT_HELP  = re.compile(r"^(help|\?|commands)$", re.I)
_INTENT_QUIT  = re.compile(r"^(quit|exit|bye|goodbye|q)$", re.I)
_INTENT_INFO  = re.compile(r"^(info|status|device|version)$", re.I)
_INTENT_OPEN  = re.compile(r"^(?:open|launch|start|run)\s+(.+)$", re.I)
_INTENT_SUM   = re.compile(r"^(?:summarize|summarise|summary of|read|explain)\s+(.+)$", re.I)
_INTENT_WIKI  = re.compile(r"^(?:search|look up|find|wiki|wikipedia)\s+(.+)$", re.I)
_INTENT_VOICE = re.compile(r"^voice\s*(on|off)$", re.I)
_INTENT_CLEAR = re.compile(r"^(clear|reset|forget|new chat)$", re.I)
_INTENT_TONE  = re.compile(r"^(be|talk|speak)\s+(casual|formal|friendly|professional).*$", re.I)

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

    # ── voice ──────────────────────────────────────────────────────────────────

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
            logger.info(f"Voice enabled -- STT: Whisper, TTS: {self._tts.backend}")
            return True
        except Exception as e:
            logger.error(f"Voice init failed: {e}")
            print(f"\n  [OPAC] Voice not available: {e}")
            print("  Install: pip install faster-whisper sounddevice pyttsx3 numpy\n")
            return False

    def enable_wake_word(self):
        if not self._stt:
            return
        try:
            from voice.wakeword import WakeWordDetector
            self._wakeword = WakeWordDetector(
                callback=self._on_wake_word,
                stt_engine=self._stt,
            )
            self._wakeword.start()
            print(f"  [OPAC] Wake word active -- say '{WAKE_WORD}' to activate\n")
        except Exception as e:
            logger.error(f"Wake word init failed: {e}")

    def _on_wake_word(self):
        """
        Called from background wake word thread.

        CRITICAL: pause the wake word mic stream FIRST, then open STT mic,
        then resume wake word. Without this, both streams compete for the
        microphone and the command audio is never captured.
        """
        if not self._stt:
            return

        # 1. Pause wake word mic so command STT gets exclusive access
        if self._wakeword:
            self._wakeword.pause()

        # Small gap so the stream fully releases the mic buffer
        import time
        time.sleep(0.3)

        # 2. Audible + visual confirmation
        self._tts_speak("Yes?")
        print("\n" + "-" * 50, flush=True)
        print("  [OPAC] Listening for your command ...", flush=True)
        print("-" * 50, flush=True)

        # 3. Listen for the actual command with a fresh mic open
        try:
            text = self._stt.listen(timeout=10.0)
        except Exception as e:
            logger.error(f"STT listen error: {e}")
            text = ""
        finally:
            # 4. Always resume wake word, even if STT failed
            time.sleep(0.2)
            if self._wakeword:
                self._wakeword.resume()

        if not text.strip():
            print("  [OPAC] Nothing heard. Say 'hey opac' again to activate.\n",
                  flush=True)
            print("  You: ", end="", flush=True)
            return

        # 5. Show what was heard
        print(f"\n  You (voice): {text}\n", flush=True)

        # 6. Route through the normal input handler
        self._handle_input(text)

        # 7. Reprint the input prompt
        print("  You: ", end="", flush=True)

    # ── Wikipedia ──────────────────────────────────────────────────────────────

    def _init_wiki(self):
        try:
            from voice.wiki import WikiEngine
            self._wiki = WikiEngine()
            self._wiki.setup()
        except Exception as e:
            logger.debug(f"Wikipedia init: {e}")

    # ── public API ─────────────────────────────────────────────────────────────

    def summarize_file(self, path: str) -> str:
        self.start()
        return self.summarizer.summarize_file(path)

    def summarize_url(self, url: str) -> str:
        self.start()
        return self.summarizer.summarize_url(url)

    def chat(self, query: str) -> str:
        """Send message to LLM. Streams response to terminal."""
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
        logger.info(f"Wikipedia context injected: topic='{topic}' ({len(results)} results)")
        return WIKI_CONTEXT_PROMPT.format(wiki_context=ctx, question=query)

    # ── interactive REPL ───────────────────────────────────────────────────────

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
            try:
                raw = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  [OPAC] Goodbye!\n")
                break

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

    def run_voice_mode(self):
        """Fully hands-free mode."""
        if not self.enable_voice():
            print("  [OPAC] Cannot start voice mode -- libraries not installed.")
            return
        self.start()
        self.enable_wake_word()
        print(f"\n  [OPAC] Voice mode active. Say '{WAKE_WORD}' to start.")
        print("  Press Ctrl+C to exit.\n")
        self._tts_speak("Voice mode active. Say hey opac to talk to me.")
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n  [OPAC] Goodbye.\n")
        finally:
            self.stop()

    # ── router ─────────────────────────────────────────────────────────────────

    def _handle_input(self, user_input: str):
        if _CASUAL.match(user_input.strip()):
            print("\n  OPAC: ", end="", flush=True)
            self.chat(user_input)
            print()
            return

        m = _INTENT_WIKI.match(user_input)
        if m:
            self._do_wiki_search(m.group(1).strip())
            return

        url_match = re.search(r"https?://\S+", user_input)
        if url_match:
            url = url_match.group(0)
            print(f"\n  [OPAC] Fetching: {url}")
            result = self.summarizer.summarize_url(url, stream=True)
            if result:
                print(f"\n  OPAC: {result}\n")
                self._tts_speak(result)
            return

        clean = user_input.strip().strip('"').strip("'")
        if _looks_like_path(clean):
            print(f"\n  [OPAC] Summarising: {clean}")
            result = self.summarizer.summarize_file(clean, stream=True)
            if result:
                print(f"\n  OPAC: {result}\n")
                self._tts_speak(result)
            return

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

        m = _INTENT_OPEN.match(user_input)
        if m:
            print(f"\n  OPAC: App launcher coming in Phase 5. (asked to open: '{m.group(1)}')\n")
            return

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
            print(f"  [OPAC] No Wikipedia results -- answering from knowledge ...\n")
            print("  OPAC: ", end="", flush=True)
            self.chat(f"Tell me about {topic}")
            print()
            return
        ctx    = self._wiki.format_context(results)
        prompt = WIKI_CONTEXT_PROMPT.format(wiki_context=ctx,
                                            question=f"Tell me about {topic}")
        print(f"  [OPAC] Found {len(results)} article(s). Generating answer ...\n")
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

    def _build_tone_system(self, user_message: str) -> str:
        if self._tone == "casual":
            hint = ("\n\nTone instruction: Be casual, warm, and friendly. "
                    "Use everyday language. Light humour is welcome.")
        elif self._tone in ("formal", "professional"):
            hint = ("\n\nTone instruction: Be precise, professional, and well-structured. "
                    "Use clear formal language.")
        else:
            hint = self._auto_detect_tone(user_message)
        return SYSTEM_PROMPT + hint

    def _auto_detect_tone(self, text: str) -> str:
        casual_re = re.compile(
            r"\b(hey|hi|lol|haha|btw|tbh|ngl|omg|cool|awesome|wanna|gonna|"
            r"gotta|kinda|sorta|ya|yep|nope|dunno|gimme|lemme|sup)\b", re.I
        )
        formal_re = re.compile(
            r"\b(please|could you|would you|kindly|regarding|furthermore|"
            r"therefore|however|in addition|specifically|approximately|"
            r"subsequently|consequently|nevertheless)\b", re.I
        )
        casual = len(casual_re.findall(text))
        formal = len(formal_re.findall(text))
        if casual > formal:
            return ("\n\nTone instruction: The user is casual -- "
                    "respond warmly and naturally, like a helpful friend.")
        elif formal > casual:
            return ("\n\nTone instruction: The user is formal -- "
                    "respond with precision and professionalism.")
        return ""

    def _tts_speak(self, text: str) -> None:
        if self._voice_active and self._tts and self._tts.loaded:
            self._tts.speak(text)

    def _print_help(self):
        print("""
  +---------------------------------------------------------------+
  |  OPAC Commands                                                |
  +---------------------------------------------------------------+
  |  <anything>               Chat freely -- ask anything         |
  |  hello / hi               Start a conversation                |
  |  <file path>              Summarise a file                    |
  |  <URL>                    Summarise a web page                |
  |  summarize <file/URL>     Summarise (keyword optional)        |
  |  search <topic>           Search Wikipedia for a topic        |
  |  voice on / voice off     Toggle spoken responses             |
  |  be casual / be formal    Change conversation tone            |
  |  clear                    Clear conversation memory           |
  |  info                     Show device and model status        |
  |  help                     Show this menu                      |
  |  quit / exit              Exit OPAC                           |
  +---------------------------------------------------------------+

  Voice: Say "hey opac" (or "opac", "hello opac") to activate.
         After the beep + "Listening..." message, speak your command.
         OPAC shows what it heard, then responds in text and speech.
""")

    def _print_status(self):
        model_ready  = "loaded" if self.engine.loaded else "not loaded"
        wiki_ok      = "available" if (self._wiki and self._wiki.available) else "not available"
        wiki_mode    = getattr(self._wiki, "_mode", "none") if self._wiki else "none"
        voice_status = "on" if self._voice_active else "off"
        print(f"""
  Device    : {self.engine.device}
  Model     : {self.engine.model_dir.name}
  Pipeline  : {model_ready}
  Voice     : {voice_status}
  Wikipedia : {wiki_ok} ({wiki_mode})
  Tone      : {self._tone}
  Memory    : {len(self._history)//2} turn(s) stored
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