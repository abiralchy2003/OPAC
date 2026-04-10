"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Listens continuously for the wake phrase "Hey OPAC" using a lightweight
keyword-spotting approach. Two methods are supported:

  1. openwakeword  — neural wake word model, very accurate, ~1% CPU
  2. Simple energy + STT fallback — works without openwakeword installed

Install openwakeword (recommended):
    pip install openwakeword

The wake word runs on CPU continuously — it does NOT touch the NPU.
The NPU only activates when actual LLM inference is needed.
"""

from __future__ import annotations
import threading
import time
from utils.logger import get_logger
from config.settings import WAKE_WORD, MIC_ENERGY_THRESHOLD

logger = get_logger("opac.voice.wakeword")


class WakeWordDetector:
    """
    Runs in a background thread and calls `callback()` when wake word detected.
    """

    def __init__(self, callback, stt_engine=None):
        self.callback   = callback
        self.stt        = stt_engine    # used for simple fallback detection
        self._running   = False
        self._thread    = None
        self._method    = None

    def start(self) -> None:
        """Start listening for wake word in background."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Wake word detector started (phrase: '{WAKE_WORD}')")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Try openwakeword first, then fall back to simple method."""
        if self._try_openwakeword():
            return
        self._run_simple_fallback()

    def _try_openwakeword(self) -> bool:
        """
        Use openwakeword for accurate, low-CPU wake word detection.
        Returns True if successfully ran (even if it exits due to stop).
        """
        try:
            import openwakeword
            from openwakeword.model import Model
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.debug("openwakeword not installed — using simple fallback")
            return False

        try:
            # Load a pre-trained model
            # openwakeword ships with several built-in models
            oww_model = Model(
                wakeword_models=["hey_jarvis"],   # closest to "hey opac"
                inference_framework="onnx"
            )
            self._method = "openwakeword"
            logger.info("Wake word: openwakeword model loaded.")

            RATE  = 16000
            CHUNK = 1280  # 80ms at 16kHz

            with sd.InputStream(samplerate=RATE, channels=1,
                                dtype="int16", blocksize=CHUNK) as stream:
                while self._running:
                    audio, _ = stream.read(CHUNK)
                    audio_np  = audio.flatten()
                    prediction = oww_model.predict(audio_np)
                    # Check if any wake word score exceeds threshold
                    for model_name, score in prediction.items():
                        if score > 0.5:
                            logger.info(f"Wake word detected! (score={score:.2f})")
                            self.callback()
                            time.sleep(1.0)   # debounce

            return True

        except Exception as e:
            logger.debug(f"openwakeword runtime error: {e}")
            return False

    def _run_simple_fallback(self) -> None:
        """
        Simple fallback: detect loud audio, then use Whisper STT to check
        if the transcription contains the wake phrase.
        This uses more CPU than openwakeword but requires no extra models.
        """
        self._method = "simple"
        logger.info(f"Wake word: simple STT fallback (listening for '{WAKE_WORD}')")

        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("sounddevice not installed: pip install sounddevice numpy")
            return

        RATE  = 16000
        CHUNK = 1024

        if not self.stt or not self.stt.loaded:
            logger.warning("STT not loaded — wake word detection unavailable")
            return

        buffer     = []
        collecting = False
        silent     = 0.0

        with sd.InputStream(samplerate=RATE, channels=1,
                            dtype="int16", blocksize=CHUNK) as stream:
            while self._running:
                data, _ = stream.read(CHUNK)
                energy  = np.abs(data).mean()

                if energy > MIC_ENERGY_THRESHOLD:
                    collecting = True
                    silent     = 0.0
                    buffer.append(data.copy())
                elif collecting:
                    buffer.append(data.copy())
                    silent += CHUNK / RATE
                    if silent >= 1.5:   # 1.5s of silence after speech
                        audio = np.concatenate(buffer, axis=0)
                        text  = self.stt._transcribe(audio).lower()
                        logger.debug(f"Wake word check: '{text}'")
                        if WAKE_WORD.lower() in text:
                            logger.info(f"Wake word detected in: '{text}'")
                            self.callback()
                            time.sleep(0.5)
                        buffer     = []
                        collecting = False
                        silent     = 0.0
