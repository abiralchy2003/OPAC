"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Fuzzy wake word matching because Whisper transcribes "hey opac" as:
  opec, o-back, oh back, heel back, feedback, any feedback, etc.
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
    "hello opec", "opac", "opec", "o-back", "oh-back",
    "heel back", "any feedback", "feedback",
]
FUZZY_THRESHOLD = 0.55


def _is_wake_word(text: str) -> bool:
    t = text.lower().strip()
    for trigger in WAKE_TRIGGERS:
        if trigger in t:
            logger.info(f"Wake word exact match: '{trigger}' in '{t}'")
            return True
    for trigger in WAKE_TRIGGERS:
        ratio = difflib.SequenceMatcher(None, t, trigger).ratio()
        if ratio >= FUZZY_THRESHOLD:
            logger.info(f"Wake word fuzzy match: '{t}' ~ '{trigger}' ({ratio:.2f})")
            return True
    for word in t.split():
        for variant in ["opac", "opec", "opak"]:
            if difflib.SequenceMatcher(None, word, variant).ratio() >= 0.70:
                logger.info(f"Wake word word match: '{word}' ~ '{variant}'")
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


class WakeWordDetector:
    def __init__(self, callback, stt_engine=None):
        self.callback = callback
        self.stt      = stt_engine
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Wake word detector started (fuzzy matching enabled)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        if self._try_openwakeword():
            return
        self._run_simple_fallback()

    def _try_openwakeword(self) -> bool:
        try:
            import openwakeword
            from openwakeword.model import Model
            import sounddevice as sd
            import numpy as np
        except ImportError:
            return False
        try:
            oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            logger.info("Wake word: openwakeword loaded.")
            RATE, CHUNK = 16000, 1280
            device = _get_input_device()
            with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                blocksize=CHUNK, device=device) as stream:
                while self._running:
                    audio, _ = stream.read(CHUNK)
                    for _, score in oww.predict(audio.flatten()).items():
                        if score > 0.5:
                            self.callback()
                            time.sleep(1.0)
            return True
        except Exception as e:
            logger.debug(f"openwakeword error: {e}")
            return False

    def _run_simple_fallback(self):
        logger.info("Wake word: STT fallback with fuzzy matching")
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
            logger.info(f"Wake word: mic open (device={device})")
        except Exception as e:
            logger.error(f"Cannot open microphone: {e}")
            return

        buffer, collecting, silent = [], False, 0.0
        try:
            while self._running:
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
                                self.callback()
                                time.sleep(0.8)
                        except Exception as e:
                            logger.debug(f"Error: {e}")
                        buffer, collecting, silent = [], False, 0.0
        finally:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass