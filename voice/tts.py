"""
OPAC Text-to-Speech  (Phase 3)
================================
Converts OPAC's text responses to speech.

Backends (auto-detected in priority order):
  1. Piper TTS   — local neural TTS, sounds natural, works on Windows + Linux
  2. Windows SAPI — built-in Windows TTS, no install needed, works offline
  3. espeak-ng   — Linux fallback, robotic but reliable

Install Piper (recommended):
    pip install piper-tts
    # Then download a voice model from:
    # https://huggingface.co/rhasspy/piper-voices/tree/main
    # Set PIPER_VOICE_MODEL in config/settings.py to the .onnx file path

Usage:
    tts = TTSEngine()
    tts.load()
    tts.speak("Hello, I am OPAC.")
"""

from __future__ import annotations
import platform
from utils.logger import get_logger
from config.settings import TTS_ENGINE, PIPER_VOICE_MODEL

logger = get_logger("opac.voice.tts")


class TTSEngine:
    def __init__(self):
        self._backend = None
        self._engine  = None
        self._loaded  = False

    def load(self) -> None:
        if self._loaded:
            return

        engine_pref = TTS_ENGINE.lower()

        if engine_pref in ("auto", "piper") and PIPER_VOICE_MODEL:
            if self._try_piper():
                return

        if engine_pref in ("auto", "sapi") and platform.system() == "Windows":
            if self._try_sapi():
                return

        if engine_pref in ("auto", "espeak"):
            if self._try_espeak():
                return

        # Last resort: print only (no audio)
        logger.warning("No TTS backend available — responses will be text only.")
        self._backend = "none"
        self._loaded  = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend or "none"

    def speak(self, text: str) -> None:
        """Convert text to speech and play through speakers."""
        if not self._loaded:
            self.load()
        if not text.strip():
            return

        # Clean text for speech — remove markdown and special chars
        clean = _clean_for_speech(text)

        if self._backend == "piper":
            self._speak_piper(clean)
        elif self._backend == "sapi":
            self._speak_sapi(clean)
        elif self._backend == "espeak":
            self._speak_espeak(clean)
        else:
            pass  # text-only mode

    # ── backend loaders ───────────────────────────────────────────────────────

    def _try_piper(self) -> bool:
        try:
            from piper import PiperVoice
            import wave, io
            self._piper_voice = PiperVoice.load(PIPER_VOICE_MODEL)
            self._backend = "piper"
            self._loaded  = True
            logger.info(f"TTS: Piper loaded ({PIPER_VOICE_MODEL})")
            return True
        except Exception as e:
            logger.debug(f"Piper not available: {e}")
            return False

    def _try_sapi(self) -> bool:
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 175)   # words per minute
            self._engine.setProperty("volume", 0.9)
            # Try to pick a better voice if available
            voices = self._engine.getProperty("voices")
            for v in voices:
                if "zira" in v.name.lower() or "david" in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break
            self._backend = "sapi"
            self._loaded  = True
            logger.info("TTS: Windows SAPI (pyttsx3) ready.")
            return True
        except Exception as e:
            logger.debug(f"SAPI not available: {e}")
            return False

    def _try_espeak(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(["espeak-ng", "--version"],
                                    capture_output=True, timeout=3)
            if result.returncode == 0:
                self._backend = "espeak"
                self._loaded  = True
                logger.info("TTS: espeak-ng ready.")
                return True
        except Exception as e:
            logger.debug(f"espeak-ng not available: {e}")
        return False

    # ── speech methods ────────────────────────────────────────────────────────

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

    def _speak_sapi(self, text: str) -> None:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            logger.error(f"SAPI speak error: {e}")

    def _speak_espeak(self, text: str) -> None:
        try:
            import subprocess
            subprocess.run(["espeak-ng", "-s", "160", text],
                           capture_output=True, timeout=60)
        except Exception as e:
            logger.error(f"espeak speak error: {e}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_for_speech(text: str) -> str:
    """Remove markdown and symbols that sound bad when spoken."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # bold
    text = re.sub(r"\*(.+?)\*",     r"\1", text)   # italic
    text = re.sub(r"`(.+?)`",       r"\1", text)   # inline code
    text = re.sub(r"#{1,6}\s*",     "",    text)   # headings
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # links
    text = re.sub(r"[-•*]\s+",      "",    text)   # bullets
    text = re.sub(r"\n{2,}",        ". ",  text)   # paragraph breaks
    text = re.sub(r"\n",            " ",   text)   # newlines
    text = re.sub(r"\s{2,}",        " ",   text)   # extra spaces
    return text.strip()
