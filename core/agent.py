"""
OPAC Agent  (Phase 1-5.5)
===========================
Phase 5.5: Voice-controlled Word, PowerPoint, Excel, Browser, Messaging
"""

from __future__ import annotations

import queue, re, threading, time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
_INTENT_TAB     = re.compile(
    r"^(?:summarize|summarise|read|what is|what'?s on|explain)\s+"
    r"(?:this\s+)?(?:tab|page|browser|current tab|current page|website|site|article)$", re.I)
_INTENT_TAB2    = re.compile(r"^(?:tab|page|browser)\s+(?:summary|summarize|summarise)$", re.I)
_INTENT_OPEN    = re.compile(r"^(?:open|launch|start|run|load)\s+(.+)$", re.I)
_INTENT_APPS    = re.compile(r"^(?:list apps?|show apps?|what apps?|available apps?).*$", re.I)

# Phase 5.5 — Office intents
_INTENT_WORD    = re.compile(r"^(?:open word|create (?:new )?(?:word )?document|new word doc).*$", re.I)
_INTENT_PPTX    = re.compile(r"^(?:open (?:power\s*point|ppt|presentation)|create (?:new )?(?:power\s*point|presentation)).*$", re.I)
_INTENT_EXCEL   = re.compile(r"^(?:open excel|create (?:new )?(?:excel|spreadsheet|workbook)).*$", re.I)
_INTENT_BROWSER = re.compile(r"^open\s+(chrome|edge|brave|firefox)\s*(.*)$", re.I)
_INTENT_VSCODE  = re.compile(r"^(?:open\s+)?(?:vs\s*code|vscode|cursor|sublime)\s*(.*)$", re.I)
_INTENT_VSCODE_NEW = re.compile(
    r"^create\s+(?:a\s+)?(?:new\s+)?(python|javascript|typescript|html|css|java|"
    r"c\+\+|cpp|c#|csharp|go|rust|ruby|php|sql|bash|powershell|markdown|json|yaml|text)\s+"
    r"file\s+(?:called|named)?\s*(.+)$", re.I)
_INTENT_NOTEPAD = re.compile(r"^open\s+notepad.*$", re.I)
_INTENT_CMD     = re.compile(r"^open\s+(?:cmd|command prompt|terminal).*$", re.I)
_INTENT_PS      = re.compile(r"^open\s+(?:powershell|ps).*$", re.I)
_EDITOR_CMDS    = re.compile(
    r"^(?:write|append|add|type|run (?:command|this)|save as|close (?:notepad|cmd|powershell))", re.I)
_VSCODE_CMDS    = re.compile(
    r"^(?:write (?:a )?(?:function|class|method|code|script)|"
    r"create (?:a )?(?:function|class)|add (?:a )?(?:function|class|import)|"
    r"run (?:the )?file|run it|save (?:the )?file|close (?:vs\s*code|vscode)|"
    r"vs\s*code status)", re.I)
_INTENT_MSG     = re.compile(
    r"^(?:send\s+)?(?:message|msg)\s+(?:to\s+)?(.+?)\s+(.+)$", re.I)
_INTENT_SEND_APP = re.compile(
    r"^(?:send\s+)?(?:via\s+|through\s+|using\s+)?(viber|telegram|discord|skype|signal)\s+(?:to\s+)?(.+?)\s+(.+)$", re.I)
_INTENT_WHATSAPP = re.compile(
    r"^(?:send\s+)?whatsapp\s+(?:to\s+)?(.+?)\s+(.+)$", re.I)

# Office session continuation commands (these are caught AFTER session is active)
_WORD_CMDS = re.compile(
    r"^(?:write|add (?:heading|subheading|paragraph|bullet|numbered|table|page break)|"
    r"create (?:heading|table)|save as|close word|word status)", re.I)
_PPTX_CMDS = re.compile(
    r"^(?:create (?:next|new)? ?slide|add slide|new slide|"
    r"create (?:bullet|section|title|blank|content) slide|"
    r"save as|close (?:power\s*point|presentation)|pptx? status)", re.I)
_EXCEL_CMDS = re.compile(
    r"^(?:add (?:header|row|total|sheet)|create sheet|save as|close excel|excel status)", re.I)
_BROWSER_CMDS = re.compile(
    r"^(?:search (?:for|on|youtube|google)?|go to|navigate to|open new tab|"
    r"new tab|read (?:this )?page|close browser|browser close)", re.I)

_CASUAL = re.compile(
    r"^(hi|hello|hey|how are you|what.?s up|good\s+(morning|afternoon|evening|night)|"
    r"howdy|sup|greetings|yo\b|morning|evening|afternoon|"
    r"how.?s (it going|everything)|what can you do|who are you|"
    r"tell me about yourself|introduce yourself|"
    r"thanks?|thank you|cheers|cool|nice|great|awesome|ok|okay|alright|"
    r"are you there|you there|can you hear me).*$", re.I)

_WANTS_LONG = re.compile(
    r"\b(explain|elaborate|tell me everything|comprehensive|in depth|"
    r"detailed|full|complete|thorough|write an essay|write a report)\b", re.I)

_TOPIC_STRIP = re.compile(
    r"^(?:can you |could you |please |do you know |tell me |"
    r"what is |what are |who is |who was |explain |describe |"
    r"give me info(?:rmation)? (?:on|about) |"
    r"tell me about |i want to know about |"
    r"i want to learn about |what do you know about )", re.I)

_SENTENCE_END = re.compile(r"([.!?])\s+")


def _length_hint(query: str) -> str:
    q = query.strip()
    if _WANTS_LONG.search(q):
        return ""
    if _CASUAL.match(q):
        return "\n\nLength: 2-3 sentences maximum. Be warm and brief."
    if len(q.split()) < 8:
        return "\n\nLength: 4-8 sentences maximum. Be concise and direct."
    return "\n\nLength: 6-8 sentences maximum. Be informative but concise."


class OPACAgent:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR,
                 device: str = INFERENCE_DEVICE,
                 model_override: str = None,
                 device_override: str = None):
        if device_override: device = device_override.upper()
        if model_override:  model_dir = DEFAULT_MODEL_DIR.parent / model_override

        self.engine     = NPUEngine(model_dir=model_dir, device=device)
        self.summarizer = Summarizer(self.engine)

        self._history: List[Dict] = []
        self._voice_active = False
        self._tone         = "auto"
        self._stt          = None
        self._tts          = None
        self._wakeword     = None
        self._voice_queue: queue.Queue = queue.Queue()

        self._browser_summary = None  # for summarize tab (Phase 4)
        self._launcher        = None  # Phase 5
        self._office          = None  # Phase 5.5

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
        if self._wakeword: self._wakeword.stop()
        if self._office:   self._office.close_all()
        self.engine.unload()

    # ── voice ──────────────────────────────────────────────────────────────────

    def enable_voice(self) -> bool:
        try:
            from voice.stt import STTEngine
            from voice.tts import TTSEngine
            if not self._stt:
                self._stt = STTEngine(); self._stt.load()
            if not self._tts:
                self._tts = TTSEngine(); self._tts.load()
            self._voice_active = True
            logger.info(f"Voice enabled -- TTS: {self._tts.backend}")
            return True
        except Exception as e:
            logger.error(f"Voice init: {e}")
            print(f"\n  [OPAC] Voice not available: {e}\n")
            return False

    def enable_wake_word(self):
        if not self._stt: return
        try:
            from voice.wakeword import WakeWordDetector
            self._wakeword = WakeWordDetector(
                callback=self._on_voice_command, stt_engine=self._stt)
            if self._tts:
                self._wakeword.set_tts_engine(self._tts)
            self._wakeword.start()
            print(f"  [OPAC] Wake word active -- say '{WAKE_WORD}' to activate\n")
        except Exception as e:
            logger.error(f"Wake word init: {e}")

    def _on_voice_command(self, text: str):
        if text and text.strip():
            if self._tts: self._tts.interrupt()
            self._voice_queue.put(text.strip())

    def _drain_voice_queue(self):
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

    # ── Wikipedia ──────────────────────────────────────────────────────────────

    def _init_wiki(self):
        try:
            from voice.wiki import WikiEngine
            self._wiki = WikiEngine(); self._wiki.setup()
        except Exception as e:
            logger.debug(f"Wikipedia init: {e}")

    # ── Office manager (lazy init) ─────────────────────────────────────────────

    def _get_office(self):
        if self._office is None:
            from actions.office import OfficeManager
            self._office = OfficeManager()
        return self._office

    # ── Phase 4 browser tab ────────────────────────────────────────────────────

    def _init_browser_summary(self):
        if self._browser_summary is None:
            from actions.browser import BrowserEngine
            self._browser_summary = BrowserEngine()

    def _summarize_current_tab(self) -> str:
        self._init_browser_summary()
        print("\n  [OPAC] Grabbing current browser tab ...", flush=True)
        try:
            url, text = self._browser_summary.get_current_tab()
        except RuntimeError as e:
            return str(e)
        if not text.strip():
            return "Page appears empty."
        if len(text) > BROWSER_MAX_CHARS:
            text = text[:BROWSER_MAX_CHARS]
        print(f"  [OPAC] Page: {url} ({len(text)} chars) — summarising ...\n")
        prompt = BROWSER_SUMMARIZE_PROMPT.format(url=url, content=text)
        print("  OPAC: ", end="", flush=True)
        collected = []; sb = []
        def _cb(tok):
            collected.append(tok); sb.append(tok)
            print(tok, end="", flush=True)
            buf = "".join(sb)
            if _SENTENCE_END.search(buf):
                parts = _SENTENCE_END.split(buf)
                for i in range(0, len(parts)-1, 2):
                    self._tts_stream(parts[i]+parts[i+1])
                sb.clear()
                if parts[-1]: sb.append(parts[-1])
            return False
        self.engine._generate_chat(user_message=prompt, streamer_callback=_cb)
        if sb: self._tts_stream("".join(sb))
        print("\n")
        response = "".join(collected).strip()
        if response:
            self._history.append({"role":"user","content":f"Summarise {url}"})
            self._history.append({"role":"assistant","content":response})
        return response

    # ── Phase 5 launcher ──────────────────────────────────────────────────────

    def _init_launcher(self):
        if self._launcher is None:
            from actions.launcher import AppLauncher
            self._launcher = AppLauncher()

    def _open_app(self, app_name: str) -> str:
        self._init_launcher()
        print(f"\n  [OPAC] Looking for '{app_name}' ...", flush=True)
        ok, msg = self._launcher.open(app_name)
        self._tts_speak(msg)
        return msg

    # ── public API ─────────────────────────────────────────────────────────────

    def summarize_file(self, path: str) -> str:
        self.start(); return self.summarizer.summarize_file(path)

    def summarize_url(self, url: str) -> str:
        self.start(); return self.summarizer.summarize_url(url)

    def chat(self, query: str) -> str:
        self.start()
        tone_system    = self._build_tone_system(query)
        enriched_query = self._enrich_with_wiki(query)
        recent         = self._history[-6:] if self._history else None
        collected = []; sb = []

        def _cb(tok):
            collected.append(tok); sb.append(tok)
            print(tok, end="", flush=True)
            buf = "".join(sb)
            if _SENTENCE_END.search(buf):
                parts = _SENTENCE_END.split(buf)
                for i in range(0, len(parts)-1, 2):
                    self._tts_stream(parts[i]+parts[i+1])
                sb.clear()
                if parts[-1]: sb.append(parts[-1])
            return False

        self.engine._generate_chat(
            user_message=enriched_query, system=tone_system,
            history=recent, streamer_callback=_cb)
        print()
        if sb: self._tts_stream("".join(sb))
        response = "".join(collected).strip()
        if response:
            self._history.append({"role":"user","content":query})
            self._history.append({"role":"assistant","content":response})
            if len(self._history) > 20: self._history = self._history[-20:]
        return response

    def _enrich_with_wiki(self, query: str) -> str:
        if not self._wiki or not self._wiki.available: return query
        if not self._wiki.is_factual_query(query):     return query
        topic   = _extract_topic(query)
        results = self._wiki.search(topic)
        if not results: return query
        ctx = self._wiki.format_context(results)
        return WIKI_CONTEXT_PROMPT.format(wiki_context=ctx, question=query)

    def _generate_text(self, prompt: str, max_words: int = 0) -> str:
        """Generate content on NPU for office docs — no streaming, just returns text."""
        self.start()
        if max_words:
            prompt += f"\n\nWrite exactly {max_words} words."
        result = self.engine._generate_chat(user_message=prompt)
        return result.strip()

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
            if result == "quit": break
            try:
                typed = self._input_with_voice_check("  You: ")
            except (EOFError, KeyboardInterrupt):
                print("\n\n  [OPAC] Goodbye!\n"); break
            if typed is None: continue
            raw = typed.strip()
            if not raw: continue
            if self._tts: self._tts.interrupt()
            if _INTENT_QUIT.match(raw):
                print("\n  OPAC: Goodbye! Take care.\n")
                self._tts_speak("Goodbye! Take care."); break
            if _INTENT_HELP.match(raw):  self._print_help();   continue
            if _INTENT_INFO.match(raw):  self._print_status(); continue
            if _INTENT_CLEAR.match(raw):
                self._history.clear()
                print("  [OPAC] Conversation memory cleared.\n"); continue
            m = _INTENT_TONE.match(raw)
            if m:
                self._tone = m.group(2).lower()
                print(f"  [OPAC] Tone: {self._tone}\n"); continue
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
        result_holder = [None]; done_event = threading.Event()
        def _read():
            try:    result_holder[0] = input(prompt)
            except: result_holder[0] = "\x03"
            finally: done_event.set()
        threading.Thread(target=_read, daemon=True).start()
        while not done_event.is_set():
            done_event.wait(timeout=0.2)
            if not self._voice_queue.empty():
                result = self._drain_voice_queue()
                if result == "quit": raise EOFError
                return None
        raw = result_holder[0]
        if raw == "\x03": raise KeyboardInterrupt
        return raw

    def run_voice_mode(self):
        if not self.enable_voice():
            print("  [OPAC] Cannot start voice mode."); return
        self.start(); self.enable_wake_word()
        print(f"\n  [OPAC] Voice mode. Say '{WAKE_WORD}' to start. Ctrl+C to exit.\n")
        self._tts_speak("Voice mode active. Say hey opac to talk to me.")
        try:
            while True:
                result = self._drain_voice_queue()
                if result == "quit": break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n  [OPAC] Goodbye.\n")
        finally: self.stop()

    # ── main router ────────────────────────────────────────────────────────────

    def _handle_input(self, user_input: str):
        txt = user_input.strip()

        # ── Casual greeting ───────────────────────────────────────────────────
        if _CASUAL.match(txt):
            print("\n  OPAC: ", end="", flush=True)
            self.chat(txt); print(); return

        # ── Phase 5.5 — Office session continuation ───────────────────────────
        office = self._get_office()

        # Active Word session — route word commands even without keyword
        if office.word and _WORD_CMDS.match(txt):
            self._handle_word_continuation(txt); return

        # Active PowerPoint session
        if office.pptx and _PPTX_CMDS.match(txt):
            self._handle_pptx_continuation(txt); return

        # Active Excel session
        if office.excel and _EXCEL_CMDS.match(txt):
            self._handle_excel_continuation(txt); return

        # Active browser session
        if office.browser and office.browser.active and _BROWSER_CMDS.match(txt):
            self._handle_browser_continuation(txt); return

        # Active VS Code session
        if office.vscode and _VSCODE_CMDS.match(txt):
            self._handle_vscode_continuation(txt); return

        # ── Phase 5.5 — VS Code new session ───────────────────────────────────
        m = _INTENT_VSCODE_NEW.match(txt)
        if m:
            language = m.group(1).lower()
            name     = m.group(2).strip()
            self.start()
            msg = self._get_office().vscode_cmd("create", "", {
                "language": language, "name": name, "folder": "documents"
            })
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        m = _INTENT_VSCODE.match(txt)
        if m:
            rest = (m.group(1) or "").strip()
            self.start()
            editor = "cursor" if "cursor" in txt.lower() else "sublime" if "sublime" in txt.lower() else "vscode"
            if rest and _looks_like_path(rest):
                msg = self._get_office().start_vscode(editor=editor, path=rest)
            else:
                msg = self._get_office().start_vscode(editor=editor)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # ── Phase 5.5 — New office sessions ───────────────────────────────────
        if _INTENT_WORD.match(txt):
            self.start()
            msg = self._get_office().start_word()
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _INTENT_PPTX.match(txt):
            self.start()
            # Extract presentation title
            title = re.sub(r"(?i)^(?:open power\s*point|create (?:new )?(?:power\s*point|presentation))\s*(?:about|on|called)?\s*", "", txt).strip()
            msg = self._get_office().start_pptx(title)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _INTENT_EXCEL.match(txt):
            self.start()
            msg = self._get_office().start_excel()
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Browser open command (Phase 5.5)
        m = _INTENT_BROWSER.match(txt)
        if m:
            browser_name = m.group(1).lower()
            rest         = m.group(2).strip()
            profile      = ""

            # Extract profile spec: "with work profile", "with profile 2"
            pm = re.search(r"with\s+(.+?)\s+profile|with\s+profile\s+(\w+)", rest, re.I)
            if pm:
                profile = (pm.group(1) or pm.group(2) or "").strip()
                rest    = re.sub(r"with\s+.+?\s+profile|with\s+profile\s+\w+", "", rest, flags=re.I).strip()

            # Detect if automation is needed (search, navigate, etc.)
            needs_automation = bool(re.search(
                r"\b(search|go to|navigate|open new tab|whatsapp|youtube)\b", rest, re.I))

            self.start()
            ok, msg = self._get_office().start_browser(
                browser_name, profile, for_automation=needs_automation)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg)

            # Handle "and search for X" in same command
            sm = re.search(r"(?:and\s+)?search\s+(?:for\s+)?(.+)$", rest, re.I)
            if sm and ok and needs_automation:
                self._handle_browser_continuation(f"search for {sm.group(1)}")
            return

        # WhatsApp shortcut
        m = _INTENT_WHATSAPP.match(txt)
        if m:
            contact, message = m.group(1).strip(), m.group(2).strip()
            self.start()
            ok, msg = self._get_office().send_message("whatsapp", contact, message)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Messaging (Viber, Telegram etc.)
        m = _INTENT_MSG.match(txt)
        if m:
            app, contact, message = m.group(1), m.group(2).strip(), m.group(3).strip()
            self.start()
            ok, msg = self._get_office().send_message(app, contact, message)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # ── Phase 4 — browser tab ─────────────────────────────────────────────
        if _INTENT_TAB.match(txt) or _INTENT_TAB2.match(txt):
            self.start(); self._summarize_current_tab(); return

        # ── Phase 5.5 — Notepad / CMD / PowerShell ───────────────────────────
        if _INTENT_NOTEPAD.match(txt):
            self.start()
            msg = self._get_office().start_editor("notepad")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _INTENT_CMD.match(txt):
            self.start()
            msg = self._get_office().start_editor("cmd")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _INTENT_PS.match(txt):
            self.start()
            msg = self._get_office().start_editor("powershell")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Editor session continuation
        office = self._get_office()
        if office.editor and _EDITOR_CMDS.match(txt):
            self._handle_editor_continuation(txt); return

        # ── Phase 5 — list apps ───────────────────────────────────────────────
        if _INTENT_APPS.match(txt):
            self._init_launcher()
            _, msg = self._launcher.list_apps()
            print(msg); return

        # ── Phase 5 — open app (non-browser) ─────────────────────────────────
        m = _INTENT_OPEN.match(txt)
        if m:
            app = m.group(1).strip()
            if not _looks_like_path(app) and not app.startswith("http"):
                self.start()
                result = self._open_app(app)
                print(f"\n  OPAC: {result}\n"); return

        # ── Wikipedia ─────────────────────────────────────────────────────────
        m = _INTENT_WIKI.match(txt)
        if m:
            self._do_wiki_search(m.group(1).strip()); return

        # ── URL ───────────────────────────────────────────────────────────────
        url_match = re.search(r"https?://\S+", txt)
        if url_match:
            url = url_match.group(0)
            print(f"\n  [OPAC] Fetching: {url}")
            result = self.summarizer.summarize_url(url, stream=True)
            if result: print(f"\n  OPAC: {result}\n"); self._tts_speak(result)
            return

        # ── File path ─────────────────────────────────────────────────────────
        clean = txt.strip('"').strip("'")
        if _looks_like_path(clean):
            print(f"\n  [OPAC] Summarising: {clean}")
            result = self.summarizer.summarize_file(clean, stream=True)
            if result: print(f"\n  OPAC: {result}\n"); self._tts_speak(result)
            return

        # ── Summarize keyword ─────────────────────────────────────────────────
        m = _INTENT_SUM.match(txt)
        if m:
            target = m.group(1).strip().strip('"').strip("'")
            if re.match(r"https?://\S+", target):
                result = self.summarizer.summarize_url(target, stream=True)
            elif _looks_like_path(target):
                result = self.summarizer.summarize_file(target, stream=True)
            else:
                result = f"Path or URL not found: {target}"
            if result: print(f"\n  OPAC: {result}\n"); self._tts_speak(result)
            return

        # ── General chat ──────────────────────────────────────────────────────
        print("\n  OPAC: ", end="", flush=True)
        self.chat(txt); print()

    # ── Phase 5.5 session continuation handlers ────────────────────────────────

    def _handle_word_continuation(self, txt: str):
        """Route a command to the active Word session."""
        office = self._get_office()
        tl     = txt.lower().strip()

        # Save
        if "save as" in tl or tl.startswith("save"):
            from actions.office import parse_save
            name, folder = parse_save(txt)
            msg = office.word_cmd("save", "", {"name": name, "folder": folder})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Close
        if re.match(r"close word|close document", tl):
            msg = office.word_cmd("close", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Heading
        if re.match(r"add heading|create heading", tl):
            content = re.sub(r"(?i)^(?:add|create)\s+heading\s*", "", txt).strip()
            msg = office.word_cmd("heading", content, {"level": 1})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Subheading
        if re.match(r"add sub\s*heading|create sub\s*heading", tl):
            content = re.sub(r"(?i)^(?:add|create)\s+sub\s*heading\s*", "", txt).strip()
            msg = office.word_cmd("heading", content, {"level": 2})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Page break
        if "page break" in tl:
            msg = office.word_cmd("page_break", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Table
        tm = re.search(r"table\s+(?:with\s+)?(\d+)\s+col(?:umn)?s?\s+(?:and\s+)?(\d+)\s+rows?", tl)
        if tm or "add table" in tl or "create table" in tl:
            rows = int(tm.group(2)) if tm else 3
            cols = int(tm.group(1)) if tm else 3
            msg  = office.word_cmd("table", "", {"rows": rows, "cols": cols})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Bullet list
        if re.match(r"add bullet|create bullet|add list", tl):
            topic = re.sub(r"(?i)^(?:add|create)\s+(?:a\s+)?(?:bullet|list)\s+(?:about|on|for)?\s*", "", txt).strip()
            prompt = f"Write a bullet-point list about: {topic}. Give 5-7 clear concise bullet points. One per line, no dashes."
            print(f"\n  [OPAC] Generating bullet list ...", flush=True)
            content = self._generate_text(prompt)
            msg = office.word_cmd("bullets", content)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Numbered list
        if re.match(r"add numbered|create numbered", tl):
            topic = re.sub(r"(?i)^(?:add|create)\s+(?:a\s+)?numbered\s+(?:list\s+)?(?:about|on|for)?\s*", "", txt).strip()
            prompt = f"Write a numbered list about: {topic}. Give 5-7 items, one per line, no numbers (just the text)."
            content = self._generate_text(prompt)
            msg = office.word_cmd("numbered", content)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Write (default — generate on NPU and add as paragraph)
        topic = re.sub(r"(?i)^(?:write|add paragraph|type|insert)\s+(?:about\s+)?", "", txt).strip()
        # Check for word count
        wc_m  = re.search(r"in\s+(\d+)\s+words?", topic, re.I)
        wc    = int(wc_m.group(1)) if wc_m else 0
        if wc_m: topic = topic[:wc_m.start()].strip()
        prompt = f"Write a paragraph about: {topic}."
        if wc:   prompt += f" Write exactly {wc} words."
        print(f"\n  [OPAC] Generating content ...", flush=True)
        content = self._generate_text(prompt, max_words=wc)
        msg = office.word_cmd("paragraph", content)
        print(f"\n  OPAC: {msg} — '{content[:60]}...'\n"); self._tts_speak(msg)

    def _handle_pptx_continuation(self, txt: str):
        """Route a command to the active PowerPoint session."""
        from actions.office import parse_slide
        office = self._get_office()
        tl     = txt.lower().strip()

        if "save as" in tl or tl.startswith("save"):
            from actions.office import parse_save
            name, folder = parse_save(txt)
            msg = office.pptx_cmd("save", "", {"name": name, "folder": folder})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"close (?:power\s*point|presentation)", tl):
            msg = office.pptx_cmd("close", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"create blank slide|add blank slide", tl):
            msg = office.pptx_cmd("blank_slide", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"create (?:a\s+)?(?:section|title)\s+slide", tl):
            title = re.sub(r"(?i)^create\s+(?:a\s+)?(?:section|title)\s+slide\s+(?:called|about|named)?\s*", "", txt).strip()
            msg = office.pptx_cmd("section_slide", title)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"create (?:a\s+)?bullet\s+slide", tl):
            slide_title, topic = parse_slide(txt)
            print(f"\n  [OPAC] Generating bullets for '{slide_title}' ...", flush=True)
            prompt  = (f"Write 5 concise bullet points about: {topic}. "
                       f"One per line, no dashes or numbers.")
            content = self._generate_text(prompt)
            msg     = office.pptx_cmd("bullet_slide", content, {"title": slide_title})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Default: content slide
        slide_title, topic = parse_slide(txt)
        print(f"\n  [OPAC] Generating content for slide '{slide_title}' ...", flush=True)
        prompt  = (f"Write 3-4 sentences of presentation slide content about: {topic}. "
                   f"Be concise and clear, suitable for a slide body.")
        content = self._generate_text(prompt)
        msg     = office.pptx_cmd("content_slide", content, {"title": slide_title})
        print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg)

    def _handle_excel_continuation(self, txt: str):
        """Route a command to the active Excel session."""
        office = self._get_office()
        tl     = txt.lower().strip()

        if "save as" in tl or tl.startswith("save"):
            from actions.office import parse_save
            name, folder = parse_save(txt)
            msg = office.excel_cmd("save", "", {"name": name, "folder": folder})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if "close excel" in tl:
            msg = office.excel_cmd("close", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"add (?:total|sum)\s*row", tl):
            msg = office.excel_cmd("total", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"create sheet|add sheet|new sheet", tl):
            name = re.sub(r"(?i)^(?:create|add|new)\s+sheet\s+(?:called|named)?\s*", "", txt).strip()
            msg  = office.excel_cmd("sheet", name)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"add header", tl):
            content = re.sub(r"(?i)^add\s+headers?\s*", "", txt).strip()
            msg = office.excel_cmd("headers", content)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"add row", tl):
            content = re.sub(r"(?i)^add\s+row\s+(?:with\s+)?", "", txt).strip()
            msg = office.excel_cmd("row", content)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

    def _handle_vscode_continuation(self, txt: str):
        """Route a command to the active VS Code session."""
        office = self._get_office()
        tl     = txt.lower().strip()

        if re.match(r"close (?:vs\s*code|vscode)", tl):
            msg = office.vscode_cmd("close", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if re.match(r"run (?:the )?file|run it", tl):
            msg = office.vscode_cmd("run", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Write function / class / code
        topic = re.sub(
            r"(?i)^(?:write|add|create)\s+(?:a\s+)?(?:function|class|method|code|script|import)\s*(?:that|which|to|called|named|for)?\s*",
            "", txt
        ).strip()
        if not topic:
            topic = txt

        # Detect what kind of code to generate
        lang = getattr(office.vscode, '_language', 'python') if office.vscode else 'python'
        if re.search(r"\bfunction\b", txt, re.I):
            prompt = f"Write a {lang} function that: {topic}. Only output the code, no explanation."
        elif re.search(r"\bclass\b", txt, re.I):
            prompt = f"Write a {lang} class for: {topic}. Only output the code, no explanation."
        elif re.search(r"\bimport\b", txt, re.I):
            prompt = f"Write the import statements needed for: {topic} in {lang}. Only output the code."
        else:
            prompt = f"Write {lang} code that: {topic}. Only output the code, no explanation."

        print(f"\n  [OPAC] Generating {lang} code ...", flush=True)
        code = self._generate_text(prompt)

        if code:
            # Strip markdown code fences if present
            import re as _re
            code = _re.sub(r"```[\w]*\n?", "", code).strip("`").strip()
            msg  = office.vscode_cmd("append", code)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg)
        else:
            print("\n  OPAC: Could not generate code.\n")

    def _handle_editor_continuation(self, txt: str):
        """Route commands to active Notepad/CMD/PowerShell session."""
        import re as _r
        office = self._get_office()
        tl = txt.lower().strip()

        if _r.match(r"close (notepad|cmd|powershell|ps)", tl):
            msg = office.editor_cmd("close", "")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if "save as" in tl or tl.startswith("save"):
            from actions.office import parse_save
            n, f_ = parse_save(txt)
            msg = office.editor_cmd("save", "", {"name": n, "folder": f_})
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _r.match(r"run (command|this|it)?", tl):
            cmd_text = _r.sub(r"(?i)^run\s+(?:command\s+)?", "", txt).strip()
            msg = office.editor_cmd("run", cmd_text)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _r.match(r"add (a )?heading", tl):
            h = _r.sub(r"(?i)^add\s+(?:a\s+)?heading\s*", "", txt).strip()
            msg = office.editor_cmd("heading", h)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _r.match(r"add (a )?(bullet|list)", tl):
            topic = _r.sub(r"(?i)^add\s+(?:a\s+)?(?:bullet\s+)?(?:list\s+)?(?:about|on|for)?\s*", "", txt).strip()
            print(f"\n  [OPAC] Generating bullets ...", flush=True)
            content = self._generate_text(
                f"Write 5 concise bullet points about: {topic}. One per line, plain text only.")
            msg = office.editor_cmd("bullets", content)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _r.match(r"append|add to", tl):
            topic = _r.sub(r"(?i)^(?:append|add\s+to)\s+(?:about\s+)?", "", txt).strip()
            m2 = _r.search(r"in (\d+) words?", topic, _r.I)
            wc = int(m2.group(1)) if m2 else 0
            if m2: topic = topic[:m2.start()].strip()
            print(f"\n  [OPAC] Generating content ...", flush=True)
            prompt = f"Write a paragraph about: {topic}."
            if wc: prompt += f" Write exactly {wc} words."
            content = self._generate_text(prompt, max_words=wc)
            msg = office.editor_cmd("append", content)
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        # Default: write — strip "write about" then generate on NPU
        topic = _r.sub(r"(?i)^(?:write|type|add)\s+(?:about\s+)?", "", txt).strip()
        m2 = _r.search(r"in (\d+) words?", topic, _r.I)
        wc = int(m2.group(1)) if m2 else 0
        if m2: topic = topic[:m2.start()].strip()
        print(f"\n  [OPAC] Generating content ...", flush=True)
        prompt = f"Write a paragraph about: {topic}."
        if wc: prompt += f" Write exactly {wc} words."
        content = self._generate_text(prompt, max_words=wc)
        es = office.editor
        if es and es._content.strip():
            msg = office.editor_cmd("append", content)
        else:
            msg = office.editor_cmd("write", content)
        print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg)

    def _handle_browser_continuation(self, txt: str):
        """Route a command to the active browser session.
        Auto-connects via CDP if no Playwright session is active."""
        import re as _re
        office = self._get_office()
        tl     = txt.lower().strip()

        # If no active Playwright session, try CDP first
        if not (office.browser and office.browser.active):
            from actions.office import BrowserSession
            b = BrowserSession()
            ok, _ = b.connect_cdp()
            if ok:
                office.browser = b
                print("  [OPAC] Connected to running browser", flush=True)
            else:
                print("\n  OPAC: No browser available. "
                      "Open a browser with 'open chrome and search X'.\n")
                return

        if _re.match(r"close browser", tl):
            _, msg = office.browser_cmd("close")
            self._tts_speak(msg); return

        m = _re.search(r"(?:search|look up)\s+(?:on\s+)?youtube\s+(?:for\s+)?(.+)$", tl, _re.I)
        if m:
            ok, msg = office.browser_cmd("search_youtube", m.group(1).strip())
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        m = _re.search(r"search\s+(?:for\s+)?(.+)$", tl, _re.I)
        if m:
            ok, msg = office.browser_cmd("search", m.group(1).strip())
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        m = _re.search(r"(?:go to|navigate to|open)\s+(.+)$", tl, _re.I)
        if m:
            ok, msg = office.browser_cmd("navigate", m.group(1).strip())
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _re.match(r"(?:open\s+)?new tab", tl):
            ok, msg = office.browser_cmd("new_tab")
            print(f"\n  OPAC: {msg}\n"); self._tts_speak(msg); return

        if _re.match(r"read (?:this )?page|what is on this page", tl):
            ok, text = office.browser_cmd("read")
            print(f"\n  OPAC: Summarising page ...\n")
            self._summarize_text(text); return

    def _do_wiki_search(self, query: str):
        if not self._wiki or not self._wiki.available:
            print("\n  OPAC: ", end="", flush=True)
            self.chat(f"Tell me about {query}"); print(); return
        topic   = _extract_topic(query)
        print(f"\n  [OPAC] Searching Wikipedia: '{topic}' ...")
        results = self._wiki.search(topic)
        if not results:
            print("  [OPAC] No results -- answering from knowledge ...\n")
            print("  OPAC: ", end="", flush=True)
            self.chat(f"Tell me about {topic}"); print(); return
        ctx    = self._wiki.format_context(results)
        prompt = WIKI_CONTEXT_PROMPT.format(wiki_context=ctx, question=f"Tell me about {topic}")
        print(f"  [OPAC] Found {len(results)} article(s) ...\n  OPAC: ", end="", flush=True)
        collected = []; sb = []
        def _cb(tok):
            collected.append(tok); sb.append(tok)
            print(tok, end="", flush=True)
            buf = "".join(sb)
            if _SENTENCE_END.search(buf):
                parts = _SENTENCE_END.split(buf)
                for i in range(0, len(parts)-1, 2): self._tts_stream(parts[i]+parts[i+1])
                sb.clear()
                if parts[-1]: sb.append(parts[-1])
            return False
        self.engine._generate_chat(user_message=prompt, streamer_callback=_cb)
        if sb: self._tts_stream("".join(sb))
        print("\n")
        response = "".join(collected).strip()
        if response:
            self._history.append({"role":"user","content":query})
            self._history.append({"role":"assistant","content":response})

    def _summarize_text(self, text: str):
        """Summarise arbitrary text inline."""
        if not text.strip(): print("  OPAC: Page is empty.\n"); return
        print("  OPAC: ", end="", flush=True)
        collected = []
        def _cb(tok):
            collected.append(tok); print(tok, end="", flush=True); return False
        self.engine._generate_chat(
            user_message=f"Summarise this in 4-6 sentences:\n\n{text[:3000]}",
            streamer_callback=_cb)
        print("\n")
        self._tts_speak("".join(collected).strip())

    # ── tone ───────────────────────────────────────────────────────────────────

    def _build_tone_system(self, msg: str) -> str:
        if self._tone == "casual":
            hint = "\n\nTone: Be casual, warm, and friendly."
        elif self._tone in ("formal", "professional"):
            hint = "\n\nTone: Be precise and professional."
        else:
            hint = self._auto_detect_tone(msg)
        return SYSTEM_PROMPT + hint + _length_hint(msg)

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

    def _tts_stream(self, sentence: str) -> None:
        if self._voice_active and self._tts and self._tts.loaded:
            self._tts.speak_streaming(sentence)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _print_help(self):
        print("""
  +---------------------------------------------------------------+
  |  OPAC Commands                                                |
  +---------------------------------------------------------------+
  |  CONVERSATION                                                 |
  |    <anything>                 Chat freely                     |
  |    be casual / be formal      Change tone                     |
  |    clear                      Clear memory                    |
  |                                                               |
  |  DOCUMENTS & WEB                                             |
  |    <file path>                Summarise a file                |
  |    <URL>                      Summarise a web page            |
  |    summarize tab              Summarise current browser tab   |
  |    search <topic>             Search Wikipedia                |
  |                                                               |
  |  WORD (Phase 5.5)                                            |
  |    open word                  Start new document              |
  |    write about X in N words   Generate and add paragraph      |
  |    add heading X              Add H1 heading                  |
  |    add subheading X           Add H2 heading                  |
  |    add bullet list about X    Generate bullets                |
  |    add table 3 columns 4 rows Insert table                    |
  |    add page break             Insert page break               |
  |    save as NAME in downloads  Save and open file              |
  |    close word                 End session                     |
  |                                                               |
  |  POWERPOINT (Phase 5.5)                                      |
  |    open powerpoint about X    Start presentation              |
  |    create next slide about X  Add content slide               |
  |    create bullet slide about X Bullet point slide             |
  |    create section slide X     Section header slide            |
  |    create blank slide         Empty slide                     |
  |    save as NAME in desktop    Save and open file              |
  |                                                               |
  |  EXCEL (Phase 5.5)                                           |
  |    open excel                 Start workbook                  |
  |    add headers Month Income   Column headers                  |
  |    add row January 5000 3000  Data row                        |
  |    add total row              Sum row                         |
  |    create sheet called NAME   New sheet                       |
  |    save as NAME in downloads  Save and open file              |
  |                                                               |
  |  BROWSER (Phase 5.5)                                         |
  |    open brave                 Open Brave browser              |
  |    open chrome with work profile  Open with profile           |
  |    search for github          Search Google                   |
  |    search youtube for X       Search YouTube                  |
  |    go to youtube.com          Navigate to URL                 |
  |    open new tab               New tab                         |
  |    close browser              Close browser                   |
  |                                                               |
  |  MESSAGING (Phase 5.5)                                       |
  |    send whatsapp to John hello  Send WhatsApp Web message     |
  |    message viber to Alice hi    Send via Viber desktop        |
  |    open telegram and message X  Send via Telegram             |
  |                                                               |
  |  APP LAUNCHER (Phase 5)                                      |
  |    open chrome / open spotify   Launch any app               |
  |    list apps                    Show all launchable apps      |
  |                                                               |
  |  VOICE                                                        |
  |    voice on / voice off         Toggle TTS                    |
  |    Say "hey opac"               Activate voice input          |
  |    Speak while OPAC talks       Interrupt speech              |
  |                                                               |
  |  info / help / quit                                           |
  +---------------------------------------------------------------+
""")

    def _print_status(self):
        wiki_ok   = "available" if (self._wiki and self._wiki.available) else "not available"
        wiki_mode = getattr(self._wiki, "_mode", "none") if self._wiki else "none"
        office_s  = self._get_office().status() if self._office else "none"
        print(f"""
  Device    : {self.engine.device}
  Model     : {self.engine.model_dir.name}
  Pipeline  : {"loaded" if self.engine.loaded else "not loaded"}
  Voice     : {"on" if self._voice_active else "off"}
  Wikipedia : {wiki_ok} ({wiki_mode})
  Office    : {office_s}
  Tone      : {self._tone}
  Memory    : {len(self._history)//2} turn(s)
""")


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_topic(query: str) -> str:
    clean = query.strip()
    noise = re.compile(
        r"[.?!]*\s*(?:god bless|amen|thank you|thanks|please|okay|ok|"
        r"alright|sure|yes|no|bye|goodbye|you know|i mean|right|yeah|"
        r"yep|hmm+|uh+|um+|that'?s? (?:it|all|good)|"
        r"if you (?:can|could|will|would)|"
        r"i (?:think|guess|hope|believe))[.?!]*\s*$", re.I)
    clean = noise.sub("", clean).strip().rstrip("?.!").strip()
    clean = _TOPIC_STRIP.sub("", clean).strip()
    m = re.match(r"^(?:about|on|regarding)\s+(.+)$", clean, re.I)
    if m: clean = m.group(1).strip()
    clean = re.sub(r"\s+(?:god|please|thank|okay|alright|amen)\b.*$", "", clean, flags=re.I).strip()
    return clean if clean else query


def _looks_like_path(text: str) -> bool:
    from pathlib import Path as P
    if re.match(r"^[A-Za-z]:[/\\]", text): return True
    if text.startswith("/"): return True
    if P(text).exists(): return True
    if "." in text and " " not in text and len(text) < 260 and not text.endswith("?"):
        ext = text.rsplit(".", 1)[-1].lower()
        if ext in ("pdf","docx","doc","pptx","ppt","xlsx","xls","txt","md","csv","html","htm"):
            return True
    return False