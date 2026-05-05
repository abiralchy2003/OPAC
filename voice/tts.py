"""
OPAC Text-to-Speech  (Phase 3)
================================
Fixes:
  1. Uses Windows SAPI directly via win32com — no pyttsx3 runAndWait() bug
  2. Supports sentence-level streaming (speak as writing happens)
  3. Supports interrupt (stop mid-speech instantly)
  4. Falls back to pyttsx3, then espeak if win32com not available

Windows SAPI flags:
  SVSFDefault         = 0
  SVSFlagsAsync       = 1   (non-blocking)
  SVSFPurgeBeforeSpeak= 2   (interrupt + clear queue)
"""

from __future__ import annotations

import platform
import re
import threading
import queue
from utils.logger import get_logger
from config.settings import TTS_ENGINE, PIPER_VOICE_MODEL

logger = get_logger("opac.voice.tts")

IS_WINDOWS = platform.system() == "Windows"


class TTSEngine:
    def __init__(self):
        self._backend  = None
        self._loaded   = False
        self._sapi     = None       # win32com SAPI object
        self._pyttsx   = None       # pyttsx3 engine (fallback)
        self._tts_queue = queue.Queue()
        self._worker   = None
        self._speaking = threading.Event()
        self._stop_flag = threading.Event()

    def load(self) -> None:
        if self._loaded:
            return

        if TTS_ENGINE.lower() not in ("none", "off"):
            if PIPER_VOICE_MODEL and self._try_piper():
                pass
            elif IS_WINDOWS and self._try_sapi_direct():
                pass
            elif IS_WINDOWS and self._try_pyttsx3():
                pass
            elif self._try_espeak():
                pass
            else:
                logger.warning("No TTS backend available")
                self._backend = "none"

        self._loaded = True

        # Start background TTS worker thread
        self._worker = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend or "none"

    def speak(self, text: str) -> None:
        """Queue text for speech. Returns immediately — speaks in background."""
        if not self._loaded or not text.strip() or self._backend == "none":
            return
        clean = _clean_for_speech(text)
        if clean:
            self._tts_queue.put(clean)

    def speak_streaming(self, sentence: str) -> None:
        """Queue a sentence for immediate streaming speech."""
        self.speak(sentence)

    def interrupt(self) -> None:
        """Stop current speech and clear queue immediately."""
        # Clear the queue
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

        # Stop current speech
        self._stop_flag.set()

        if self._backend == "sapi_direct" and self._sapi:
            try:
                # SVSFPurgeBeforeSpeak = 2 — stops and clears
                self._sapi.Speak("", 2)
            except Exception:
                pass
        elif self._backend == "pyttsx3" and self._pyttsx:
            try:
                self._pyttsx.stop()
            except Exception:
                pass

        self._stop_flag.clear()

    # ── worker thread ──────────────────────────────────────────────────────────

    def _tts_worker(self):
        """Background thread — processes TTS queue continuously."""
        while True:
            try:
                text = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # shutdown signal
                break

            self._speaking.set()
            try:
                self._speak_now(text)
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
            finally:
                self._speaking.clear()
                self._tts_queue.task_done()

    def _speak_now(self, text: str) -> None:
        """Actually speak — called from worker thread."""
        if self._stop_flag.is_set():
            return

        if self._backend == "sapi_direct":
            self._speak_sapi_direct(text)
        elif self._backend == "pyttsx3":
            self._speak_pyttsx3(text)
        elif self._backend == "piper":
            self._speak_piper(text)
        elif self._backend == "espeak":
            self._speak_espeak(text)

    # ── backend loaders ────────────────────────────────────────────────────────

    def _try_sapi_direct(self) -> bool:
        """Windows SAPI via win32com — most reliable, no runAndWait() bug."""
        try:
            import win32com.client
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
            # Test it works
            sapi.Rate  = 1    # slightly faster than default
            sapi.Volume = 90

            # Try to find a good voice
            voices = sapi.GetVoices()
            for i in range(voices.Count):
                v = voices.Item(i)
                name = v.GetDescription().lower()
                if "zira" in name or "david" in name or "mark" in name:
                    sapi.Voice = v
                    break

            self._sapi    = sapi
            self._backend = "sapi_direct"
            self._loaded  = True
            logger.info("TTS: Windows SAPI (win32com) ready — supports interrupt")
            return True
        except Exception as e:
            logger.debug(f"SAPI direct failed: {e}")
            return False

    def _try_pyttsx3(self) -> bool:
        """pyttsx3 fallback — has runAndWait bug but better than nothing."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            voices = engine.getProperty("voices")
            for v in voices:
                if "zira" in v.name.lower() or "david" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
            self._pyttsx  = engine
            self._backend = "pyttsx3"
            self._loaded  = True
            logger.info("TTS: pyttsx3 ready (interrupt support limited)")
            return True
        except Exception as e:
            logger.debug(f"pyttsx3 failed: {e}")
            return False

    def _try_piper(self) -> bool:
        try:
            from piper import PiperVoice
            self._piper_voice = PiperVoice.load(PIPER_VOICE_MODEL)
            self._backend     = "piper"
            self._loaded      = True
            logger.info(f"TTS: Piper ready ({PIPER_VOICE_MODEL})")
            return True
        except Exception as e:
            logger.debug(f"Piper failed: {e}")
            return False

    def _try_espeak(self) -> bool:
        try:
            import subprocess
            r = subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=3)
            if r.returncode == 0:
                self._backend = "espeak"
                self._loaded  = True
                logger.info("TTS: espeak-ng ready")
                return True
        except Exception as e:
            logger.debug(f"espeak failed: {e}")
        return False

    # ── speech methods ─────────────────────────────────────────────────────────

    def _speak_sapi_direct(self, text: str) -> None:
        try:
            # SVSFlagsAsync = 1 — non-blocking so we can check stop_flag
            self._sapi.Speak(text, 1)
            # Wait for speech to finish, checking stop_flag
            import time
            while self._sapi.Status.RunningState == 2:  # 2 = running
                if self._stop_flag.is_set():
                    self._sapi.Speak("", 2)  # purge
                    break
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"SAPI speak error: {e}")

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            # Reinitialize engine each call to avoid runAndWait() bug
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logger.error(f"pyttsx3 speak error: {e}")

    def _speak_piper(self, text: str) -> None:
        try:
            import sounddevice as sd
            import numpy as np
            import io, wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._piper_voice.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf) as wf:
                frames = wf.readframes(wf.getnframes())
                rate   = wf.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=rate, blocking=True)
        except Exception as e:
            logger.error(f"Piper speak error: {e}")

    def _speak_espeak(self, text: str) -> None:
        try:
            import subprocess
            subprocess.run(["espeak-ng", "-s", "160", text],
                           capture_output=True, timeout=60)
        except Exception as e:
            logger.error(f"espeak error: {e}")


def _clean_for_speech(text: str) -> str:
    """Remove markdown and symbols that sound bad when spoken."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",       r"\1", text)
    text = re.sub(r"#{1,6}\s*",     "",    text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"[-•*]\s+",      "",    text)
    text = re.sub(r"\n{2,}",        ". ",  text)
    text = re.sub(r"\n",            " ",   text)
    text = re.sub(r"\s{2,}",        " ",   text)
    return text.strip()