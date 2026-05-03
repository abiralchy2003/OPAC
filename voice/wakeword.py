"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Single microphone stream architecture:
  - ONE stream reads mic audio continuously
  - Wake word detection runs on that stream
  - When wake word detected, the same stream is handed to the
    command listener — NO second stream opened, NO mic conflict

This fixes the Windows exclusive-mode mic conflict where
pause + reopen caused stt.listen() to fail silently.
"""

from __future__ import annotations

import difflib
import threading
import time
import numpy as np
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

RATE  = 16000
CHUNK = 1024


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
    print("  [OPAC] Wake word detected! Speak your command ...", flush=True)
    print("-" * 50, flush=True)
    try:
        import winsound
        winsound.Beep(880, 200)
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


def _transcribe(model, audio: np.ndarray) -> str:
    """Transcribe int16 numpy array using the loaded Whisper model."""
    if audio is None or len(audio) < RATE * 0.3:
        return ""
    audio_f32 = audio.flatten().astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_f32, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


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
        logger.info("Wake word detector started (single-stream mode)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # no-ops kept for API compatibility — no longer needed
    def pause(self): pass
    def resume(self): pass

    def _run(self):
        if self._try_openwakeword():
            return
        self._run_single_stream()

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
            with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                blocksize=1280, device=device) as stream:
                while self._running:
                    audio, _ = stream.read(1280)
                    for _, score in oww.predict(audio.flatten()).items():
                        if score > 0.5:
                            _alert()
                            self._listen_for_command(stream)
                            time.sleep(0.5)
            return True
        except Exception as e:
            logger.debug(f"openwakeword error: {e}")
            return False

    def _run_single_stream(self):
        """
        Single stream loop:
          1. Continuously read mic audio
          2. On speech burst: transcribe and check for wake word
          3. On wake word: immediately listen for command on SAME stream
          4. Pass command text to callback
        """
        logger.info("Wake word: single-stream STT fallback")

        if not self.stt or not self.stt.loaded:
            logger.warning("STT not loaded")
            return

        try:
            import sounddevice as sd
        except ImportError:
            logger.error("pip install sounddevice")
            return

        device = _get_input_device()
        try:
            stream = sd.InputStream(
                samplerate=RATE, channels=1, dtype="int16",
                blocksize=CHUNK, device=device,
            )
            stream.start()
            logger.info(f"Wake word: single mic stream open (device={device})")
        except Exception as e:
            logger.error(f"Cannot open microphone: {e}")
            return

        wake_buf, collecting, silent = [], False, 0.0

        try:
            while self._running:
                try:
                    data, _ = stream.read(CHUNK)
                    energy  = float(abs(data).mean())
                except Exception:
                    time.sleep(0.05)
                    continue

                if energy > MIC_ENERGY_THRESHOLD:
                    collecting, silent = True, 0.0
                    wake_buf.append(data.copy())
                elif collecting:
                    wake_buf.append(data.copy())
                    silent += CHUNK / RATE
                    if silent >= 1.2:
                        # Transcribe and check
                        audio = np.concatenate(wake_buf, axis=0)
                        text  = _transcribe(self.stt._model, audio).strip()
                        if text:
                            logger.debug(f"Wake check: '{text}'")
                        if text and _is_wake_word(text):
                            _alert()
                            # Listen for command on the SAME open stream
                            cmd = self._listen_for_command(stream)
                            if cmd.strip():
                                logger.info(f"Voice command: '{cmd}'")
                                self.callback(cmd)
                            time.sleep(0.5)
                        wake_buf, collecting, silent = [], False, 0.0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _listen_for_command(self, stream) -> str:
        """
        Listen for a command using the already-open stream.
        Shows live partial transcriptions so user sees what is heard.
        Returns the full transcribed command string.
        """
        if not self.stt or not self.stt._model:
            return ""

        print("  [OPAC] You said: ", end="", flush=True)

        full_buf  = []
        cmd_buf   = []
        collecting = False
        silent     = 0.0
        partials   = []
        deadline   = time.time() + 10.0   # 10 second max

        while time.time() < deadline:
            try:
                data, _ = stream.read(CHUNK)
                energy  = float(abs(data).mean())
            except Exception:
                time.sleep(0.05)
                continue

            if energy > MIC_ENERGY_THRESHOLD:
                collecting, silent = True, 0.0
                cmd_buf.append(data.copy())
                full_buf.append(data.copy())
            elif collecting:
                cmd_buf.append(data.copy())
                full_buf.append(data.copy())
                silent += CHUNK / RATE

                # Every 0.6 s of silence — show partial
                if silent >= 0.6:
                    burst = np.concatenate(cmd_buf, axis=0)
                    partial = _transcribe(self.stt._model, burst).strip()
                    if partial:
                        partials.append(partial)
                        print(partial + " ", end="", flush=True)
                    cmd_buf    = []
                    collecting = False
                    silent     = 0.0

                # 2.5 s of silence — done
                if silent >= 2.5:
                    break

        print()  # newline after live output

        if not full_buf:
            return ""

        # Final transcription on complete audio (more accurate)
        full_audio = np.concatenate(full_buf, axis=0)
        final      = _transcribe(self.stt._model, full_audio).strip()

        if final:
            live = " ".join(partials)
            if final.lower().strip() != live.lower().strip():
                print(f"  [OPAC] (heard) {final}", flush=True)
            else:
                logger.info(f"Command: '{final}'")

        return final
