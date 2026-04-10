"""
OPAC Speech-to-Text  (Phase 3)
================================
Converts spoken audio to text using faster-whisper (CPU).
The NPU stays free for LLM inference the whole time.

Install:
    pip install faster-whisper sounddevice numpy

Usage:
    stt = STTEngine()
    stt.load()
    text = stt.listen()
"""

from __future__ import annotations
import time
from utils.logger import get_logger
from config.settings import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE,
    MIC_SILENCE_TIMEOUT, MIC_ENERGY_THRESHOLD
)

logger = get_logger("opac.voice.stt")


class STTEngine:
    def __init__(self):
        self._model  = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed.\n"
                "Run: pip install faster-whisper sounddevice numpy"
            )
        logger.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ...")
        self._model  = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type="int8"
        )
        self._loaded = True
        logger.info("Whisper STT ready.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    def listen(self, timeout: float = 10.0) -> str:
        """
        Record from microphone until MIC_SILENCE_TIMEOUT seconds of silence,
        then transcribe with Whisper.
        Returns transcribed string, or "" if nothing heard.
        """
        if not self._loaded:
            raise RuntimeError("Call stt.load() first.")
        audio = self._record(timeout)
        if audio is None:
            return ""
        return self._transcribe(audio)

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file directly (for testing)."""
        if not self._loaded:
            raise RuntimeError("Call stt.load() first.")
        segments, _ = self._model.transcribe(path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── internal ──────────────────────────────────────────────────────────────

    def _record(self, timeout: float):
        """Record microphone audio. Returns numpy array or None."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError("pip install sounddevice numpy")

        RATE   = 16000
        CHUNK  = 1024
        frames = []
        silent = 0.0
        speaking = False

        print("  [OPAC] Listening ...", end="", flush=True)

        with sd.InputStream(samplerate=RATE, channels=1,
                            dtype="int16", blocksize=CHUNK) as stream:
            deadline = time.time() + timeout
            while time.time() < deadline:
                data, _ = stream.read(CHUNK)
                energy  = abs(data).mean() if hasattr(data, "mean") else 0

                if energy > MIC_ENERGY_THRESHOLD:
                    speaking = True
                    silent   = 0.0
                    frames.append(data.copy())
                elif speaking:
                    frames.append(data.copy())
                    silent += CHUNK / RATE
                    if silent >= MIC_SILENCE_TIMEOUT:
                        break

        if not frames:
            print(" (nothing heard)")
            return None

        import numpy as np
        audio = np.concatenate(frames, axis=0)
        duration = len(audio) / RATE
        print(f" ({duration:.1f}s)")
        return audio

    def _transcribe(self, audio) -> str:
        import numpy as np
        audio_f32 = audio.flatten().astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_f32, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"STT: '{text}'")
        return text
