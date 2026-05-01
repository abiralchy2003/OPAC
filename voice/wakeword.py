"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Key fix: wake word mic stream is PAUSED before command STT listens,
then RESUMED after. Prevents mic conflict where wake word stream
consumes audio meant for the user's command.
"""

from __future__ import annotations

import difflib
import threading
import time
from utils.logger import get_logger
from config.settings import MIC_ENERGY_THRESHOLD

logger = get_logger("opac.voice.wakeword")

WAKE_TRIGGERS = [
    "hey opac", "hey opec", "hey o-back", "hey o back", "hey oh back",
    "hey oh-back", "hey opak", "hi opac", "hi opec", "hello opac",
    "hello opec", "opac", "opec",
]
FUZZY_THRESHOLD      = 0.72
WORD_FUZZY_THRESHOLD = 0.80


def _is_wake_word(text: str) -> bool:
    t = text.lower().strip().rstrip(".")
    if len(t.split()) > 6:
        return False
    for trigger in WAKE_TRIGGERS:
        if trigger in t:
            logger.info(f"Wake word exact: '{trigger}' in '{t}'")
            return True
    for trigger in WAKE_TRIGGERS:
        ratio = difflib.SequenceMatcher(None, t, trigger).ratio()
        if ratio >= FUZZY_THRESHOLD:
            logger.info(f"Wake word fuzzy: '{t}' ~ '{trigger}' ({ratio:.2f})")
            return True
    if len(t.split()) <= 3:
        for word in t.split():
            word = word.strip(".,!?")
            for variant in ["opac", "opec", "opak"]:
                if difflib.SequenceMatcher(None, word, variant).ratio() >= WORD_FUZZY_THRESHOLD:
                    logger.info(f"Wake word word: '{word}' ~ '{variant}'")
                    return True
    return False


def _get_input_device():
    try:
        import sounddevice as sd
        sd.query_devices(kind='input')
        return None
    except Exception:
        pass
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d.get('max_input_channels', 0) > 0:
                return i
    except Exception:
        pass
    return None


def _alert():
    print("\n" + "-" * 50, flush=True)
    print("  [OPAC] Listening for your command ...", flush=True)
    print("-" * 50, flush=True)
    try:
        import winsound
        winsound.Beep(880, 200)
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


class WakeWordDetector:
    def __init__(self, callback, stt_engine=None):
        self.callback = callback
        self.stt      = stt_engine
        self._running = False
        self._thread  = None
        self._stream  = None
        self._paused  = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Wake word detector started (threshold=0.72)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def pause(self):
        """Stop reading mic so command STT can use it exclusively."""
        self._paused = True
        logger.debug("Wake word paused")

    def resume(self):
        """Resume wake word listening after command STT finishes."""
        self._paused = False
        logger.debug("Wake word resumed")

    def _run(self):
        if self._try_openwakeword():
            return
        self._run_simple_fallback()

    def _try_openwakeword(self) -> bool:
        try:
            import openwakeword
            from openwakeword.model import Model
            import sounddevice as sd
        except ImportError:
            return False
        try:
            oww    = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            device = _get_input_device()
            with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                                blocksize=1280, device=device) as stream:
                self._stream = stream
                while self._running:
                    if self._paused:
                        time.sleep(0.05)
                        continue
                    audio, _ = stream.read(1280)
                    for _, score in oww.predict(audio.flatten()).items():
                        if score > 0.5:
                            _alert()
                            self.callback()
                            time.sleep(1.0)
            return True
        except Exception as e:
            logger.debug(f"openwakeword error: {e}")
            return False

    def _run_simple_fallback(self):
        logger.info("Wake word: STT fallback (threshold=0.72)")
        if not self.stt or not self.stt.loaded:
            logger.warning("STT not loaded")
            return
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("pip install sounddevice numpy")
            return

        device = _get_input_device()
        RATE, CHUNK = 16000, 1024

        try:
            stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                    blocksize=CHUNK, device=device)
            stream.start()
            self._stream = stream
            logger.info(f"Wake word: mic open (device={device})")
        except Exception as e:
            logger.error(f"Cannot open microphone: {e}")
            return

        buffer, collecting, silent = [], False, 0.0
        try:
            while self._running:
                if self._paused:
                    # Clear buffer while paused — don't process stale audio
                    buffer, collecting, silent = [], False, 0.0
                    time.sleep(0.05)
                    continue

                try:
                    data, _ = stream.read(CHUNK)
                    energy  = float(abs(data).mean())
                except Exception:
                    time.sleep(0.1)
                    continue

                if energy > MIC_ENERGY_THRESHOLD:
                    collecting, silent = True, 0.0
                    buffer.append(data.copy())
                elif collecting:
                    buffer.append(data.copy())
                    silent += CHUNK / RATE
                    if silent >= 1.5:
                        try:
                            audio = np.concatenate(buffer, axis=0)
                            text  = self.stt._transcribe(audio).strip()
                            logger.debug(f"Wake check: '{text}'")
                            if _is_wake_word(text):
                                _alert()
                                self.callback()
                                time.sleep(1.5)
                        except Exception as e:
                            logger.debug(f"Wake error: {e}")
                        buffer, collecting, silent = [], False, 0.0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass