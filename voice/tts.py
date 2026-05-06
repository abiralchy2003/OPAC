"""
OPAC Text-to-Speech  (Phase 3)
================================
Sets tts_state.speaking flag while audio plays so the wake word
detector can ignore microphone input during that window — preventing
OPAC from hearing its own voice and responding to itself.

Backends tried in order:
  1. Windows SAPI via win32com  (pip install pywin32)
  2. pyttsx3 fallback           (pip install pyttsx3)
  3. espeak-ng                  (Linux)
"""

from __future__ import annotations

import platform
import queue
import re
import threading
from utils.logger import get_logger
from config.settings import TTS_ENGINE, PIPER_VOICE_MODEL

logger = get_logger("opac.voice.tts")
IS_WINDOWS = platform.system() == "Windows"

# Import shared speaking flag
try:
    from voice.tts_state import speaking as _speaking_flag
except ImportError:
    import threading as _t
    _speaking_flag = _t.Event()


class TTSEngine:
    def __init__(self):
        self._backend   = None
        self._loaded    = False
        self._sapi      = None
        self._pyttsx    = None
        self._tts_queue = queue.Queue()
        self._worker    = None
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
        self._worker = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend or "none"

    def speak(self, text: str) -> None:
        """Queue text for speech. Non-blocking."""
        if not self._loaded or not text.strip() or self._backend == "none":
            return
        clean = _clean_for_speech(text)
        if clean:
            self._tts_queue.put(clean)

    def speak_streaming(self, sentence: str) -> None:
        """Queue a sentence for immediate streaming TTS."""
        self.speak(sentence)

    def interrupt(self) -> None:
        """Stop current speech and clear queue (called by human input)."""
        # Clear queue
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        self._stop_flag.set()
        if self._backend == "sapi_direct" and self._sapi:
            try:
                self._sapi.Speak("", 2)  # SVSFPurgeBeforeSpeak
            except Exception:
                pass
        elif self._backend == "pyttsx3" and self._pyttsx:
            try:
                self._pyttsx.stop()
            except Exception:
                pass
        self._stop_flag.clear()
        _speaking_flag.clear()  # immediately mark as not speaking

    # ── worker ────────────────────────────────────────────────────────────────

    def _tts_worker(self):
        while True:
            try:
                text = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            # Set flag BEFORE speaking so mic is muted during playback
            _speaking_flag.set()
            try:
                if not self._stop_flag.is_set():
                    self._speak_now(text)
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                # Small cooldown after speech ends so mic doesn't catch
                # the tail end of audio reverb from speakers
                if self._tts_queue.empty():
                    import time
                    time.sleep(0.4)
                    _speaking_flag.clear()
                self._tts_queue.task_done()

    def _speak_now(self, text: str) -> None:
        if self._backend == "sapi_direct":
            self._speak_sapi_direct(text)
        elif self._backend == "pyttsx3":
            self._speak_pyttsx3(text)
        elif self._backend == "piper":
            self._speak_piper(text)
        elif self._backend == "espeak":
            self._speak_espeak(text)

    # ── backends ──────────────────────────────────────────────────────────────

    def _try_sapi_direct(self) -> bool:
        try:
            import win32com.client
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
            sapi.Rate   = 1
            sapi.Volume = 90
            voices = sapi.GetVoices()
            for i in range(voices.Count):
                v    = voices.Item(i)
                name = v.GetDescription().lower()
                if any(n in name for n in ("zira", "david", "mark", "hazel")):
                    sapi.Voice = v
                    break
            self._sapi    = sapi
            self._backend = "sapi_direct"
            self._loaded  = True
            logger.info("TTS: Windows SAPI (win32com) ready")
            return True
        except Exception as e:
            logger.debug(f"SAPI direct failed: {e}")
            return False

    def _try_pyttsx3(self) -> bool:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            voices = engine.getProperty("voices")
            for v in voices:
                if any(n in v.name.lower() for n in ("zira", "david")):
                    engine.setProperty("voice", v.id)
                    break
            self._pyttsx  = engine
            self._backend = "pyttsx3"
            self._loaded  = True
            logger.info("TTS: pyttsx3 ready")
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
            logger.info(f"TTS: Piper ready")
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

    # ── speech ────────────────────────────────────────────────────────────────

    def _speak_sapi_direct(self, text: str) -> None:
        try:
            import time
            self._sapi.Speak(text, 1)  # SVSFlagsAsync
            while self._sapi.Status.RunningState == 2:
                if self._stop_flag.is_set():
                    self._sapi.Speak("", 2)
                    break
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"SAPI speak error: {e}")

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")

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
            logger.error(f"Piper error: {e}")

    def _speak_espeak(self, text: str) -> None:
        try:
            import subprocess
            subprocess.run(["espeak-ng", "-s", "160", text],
                           capture_output=True, timeout=60)
        except Exception as e:
            logger.error(f"espeak error: {e}")


def _clean_for_speech(text: str) -> str:
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