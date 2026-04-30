"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Fuzzy matching for "hey opac" since Whisper transcribes it as:
  opec, o-back, oh back, heel back, here we're back, etc.

Threshold raised to 0.72 to eliminate false positives like
"mic open" -> "hi opec".

After wake word triggers:
  - Prints a clear visual indicator
  - Plays a beep (Windows) or prints bell char
  - Then listens for the actual command
"""

from __future__ import annotations

import difflib
import threading
import time
from utils.logger import get_logger
from config.settings import MIC_ENERGY_THRESHOLD

logger = get_logger("opac.voice.wakeword")

# Known Whisper transcriptions of "hey opac"
WAKE_TRIGGERS = [
    "hey opac", "hey opec", "hey o-back", "hey o back", "hey oh back",
    "hey oh-back", "hey opak", "hi opac", "hi opec", "hello opac",
    "hello opec", "opac", "opec",
]

# Raised from 0.55 to 0.72 — eliminates false positives
# "mic open" vs "hi opec" = 0.67 → no longer triggers
# "hello opec" vs "hello opac" = 0.92 → still triggers
# "he'll be back" vs "hey o-back" = 0.74 → still triggers
FUZZY_THRESHOLD = 0.72

# Words that must NOT trigger the wake word even if they match
# (safety guard for common words that happen to be similar)
EXCLUSION_PATTERNS = [
    "mic open", "microphone", "music", "movie", "become",
]


def _is_wake_word(text: str) -> bool:
    """Return True if text is likely the wake phrase."""
    t = text.lower().strip().rstrip(".")

    # Exclusion check — some common phrases score high but are not wake words
    for excl in EXCLUSION_PATTERNS:
        if excl in t:
            logger.debug(f"Wake word excluded: '{t}' contains '{excl}'")
            return False

    # Must be short enough to be a wake phrase (not a full sentence)
    # Real commands go AFTER the wake word, not during it
    if len(t.split()) > 6:
        return False

    # Exact substring match
    for trigger in WAKE_TRIGGERS:
        if trigger in t:
            logger.info(f"Wake word exact: '{trigger}' in '{t}'")
            return True

    # Fuzzy match against full phrase
    for trigger in WAKE_TRIGGERS:
        ratio = difflib.SequenceMatcher(None, t, trigger).ratio()
        if ratio >= FUZZY_THRESHOLD:
            logger.info(f"Wake word fuzzy: '{t}' ~ '{trigger}' ({ratio:.2f})")
            return True

    # Word-level: any single word similar to "opac"/"opec"
    # Only if the utterance is 1-3 words (pure wake word, not a sentence)
    if len(t.split()) <= 3:
        for word in t.split():
            word = word.strip(".,!?")
            for variant in ["opac", "opec", "opak"]:
                ratio = difflib.SequenceMatcher(None, word, variant).ratio()
                if ratio >= 0.80:  # stricter for word-level
                    logger.info(f"Wake word word-level: '{word}' ~ '{variant}' ({ratio:.2f})")
                    return True

    return False


def _get_input_device():
    """Find working microphone. Returns None for system default."""
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
    """Visual + audio alert that wake word was detected and OPAC is listening."""
    print("\n" + "=" * 50)
    print("  [OPAC] Wake word detected! Listening for command ...")
    print("=" * 50)
    # Try to play a beep
    try:
        import winsound
        winsound.Beep(880, 200)   # 880Hz for 200ms
    except Exception:
        try:
            print("\a", end="", flush=True)  # terminal bell
        except Exception:
            pass


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
        logger.info("Wake word detector started (threshold=0.72)")

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
            logger.warning("STT not loaded — wake word unavailable")
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
                                _alert()
                                self.callback()
                                time.sleep(1.5)  # debounce — don't re-trigger immediately
                        except Exception as e:
                            logger.debug(f"Wake error: {e}")
                        buffer, collecting, silent = [], False, 0.0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass