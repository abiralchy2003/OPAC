"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Self-interruption fix:
  - While TTS is speaking, mic is monitored but NOT transcribed
  - If human speaks LOUDLY (energy > INTERRUPT_THRESHOLD), TTS is
    interrupted immediately, then we listen for the command
  - OPAC's own speaker output is quieter than a human speaking
    directly into the mic, so the threshold separates them

Human interrupts OPAC:  speak loudly -> energy spike -> interrupt + listen
OPAC's own voice:       lower energy via mic -> ignored
"""

from __future__ import annotations

import difflib
import threading
import time
import numpy as np
from utils.logger import get_logger
from config.settings import MIC_ENERGY_THRESHOLD

logger = get_logger("opac.voice.wakeword")

# Import shared TTS speaking flag
try:
    from voice.tts_state import speaking as _tts_speaking
except ImportError:
    import threading as _t
    _tts_speaking = _t.Event()

WAKE_TRIGGERS = [
    "hey opac", "hey opec", "hey o-back", "hey o back", "hey oh back",
    "hey oh-back", "hey opak", "hi opac", "hi opec", "hello opac",
    "hello opec", "opac", "opec", "hello", "cook", "let's cook",
]
FUZZY_THRESHOLD      = 0.72
WORD_FUZZY_THRESHOLD = 0.80

RATE  = 16000
CHUNK = 1024

# Energy threshold to interrupt TTS — set higher than normal so OPAC's
# own speaker output (which reaches the mic at lower volume) doesn't trigger.
# Human voice directly into mic is typically 800-3000+ energy units.
# Speaker bleed-through is typically 100-400.
# Set to 600 — adjust down if your mic is far from speakers.
INTERRUPT_THRESHOLD = 600

FOLLOWUP_TIMEOUT = 10800.0  # 3 hours


def _is_wake_word(text: str) -> bool:
    t = text.lower().strip().rstrip(".!")
    if len(t.split()) > 5:
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
            word = word.strip(".,!?'")
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
    print("  [OPAC] Listening ...", flush=True)
    print("-" * 50, flush=True)
    try:
        import winsound
        winsound.Beep(880, 150)
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


def _transcribe(model, audio: np.ndarray) -> str:
    if audio is None or len(audio) < RATE * 0.3:
        return ""
    audio_f32 = audio.flatten().astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_f32, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


class WakeWordDetector:
    def __init__(self, callback, stt_engine=None):
        self.callback  = callback
        self.stt       = stt_engine
        self._running  = False
        self._thread   = None
        # Reference to TTS engine for interrupt calls
        self._tts_engine = None

    def set_tts_engine(self, tts):
        """Give the detector a reference to TTS so it can interrupt it."""
        self._tts_engine = tts

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Wake word detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

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
                    energy   = float(abs(audio).mean())

                    # While TTS is speaking, only interrupt on loud human voice
                    if _tts_speaking.is_set():
                        if energy > INTERRUPT_THRESHOLD:
                            self._interrupt_and_listen(stream)
                        continue

                    for _, score in oww.predict(audio.flatten()).items():
                        if score > 0.5:
                            _alert()
                            cmd = self._listen_for_command(stream)
                            if cmd.strip():
                                self.callback(cmd)
                                self._followup_loop(stream)
                            time.sleep(0.3)
            return True
        except Exception as e:
            logger.debug(f"openwakeword error: {e}")
            return False

    def _run_single_stream(self):
        logger.info("Wake word: single-stream STT")

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
            logger.info(f"Wake word: mic open (device={device})")
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

                # ── While TTS is speaking ──────────────────────────────────
                # Don't transcribe (would hear OPAC's own voice)
                # BUT do allow human to interrupt with a loud voice
                if _tts_speaking.is_set():
                    if energy > INTERRUPT_THRESHOLD:
                        logger.info(
                            f"Human interruption detected (energy={energy:.0f})"
                        )
                        self._interrupt_and_listen(stream)
                    else:
                        # Clear buffer so stale audio doesn't affect next wake check
                        wake_buf, collecting, silent = [], False, 0.0
                    continue

                # ── Normal wake word detection ─────────────────────────────
                if energy > MIC_ENERGY_THRESHOLD:
                    collecting, silent = True, 0.0
                    wake_buf.append(data.copy())
                elif collecting:
                    wake_buf.append(data.copy())
                    silent += CHUNK / RATE
                    if silent >= 1.2:
                        audio = np.concatenate(wake_buf, axis=0)
                        text  = _transcribe(self.stt._model, audio).strip()
                        if text:
                            logger.debug(f"Wake check: '{text}'")
                        if text and _is_wake_word(text):
                            _alert()
                            cmd = self._listen_for_command(stream)
                            if cmd.strip():
                                logger.info(f"Voice command: '{cmd}'")
                                self.callback(cmd)
                                self._followup_loop(stream)
                            time.sleep(0.3)
                        wake_buf, collecting, silent = [], False, 0.0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _interrupt_and_listen(self, stream):
        """
        Human spoke while OPAC was talking.
        Interrupt TTS immediately, then listen for the command.
        """
        # Stop TTS
        if self._tts_engine:
            self._tts_engine.interrupt()
        elif _tts_speaking.is_set():
            _tts_speaking.clear()

        print("\n  [OPAC] (interrupted)", flush=True)
        _alert()

        # Brief pause so speaker echo dies down
        time.sleep(0.3)

        cmd = self._listen_for_command(stream)
        if cmd.strip():
            logger.info(f"Interrupt command: '{cmd}'")
            self.callback(cmd)
            self._followup_loop(stream)

    def _followup_loop(self, stream):
        """Stay in listening mode for FOLLOWUP_TIMEOUT seconds."""
        print("\n  [OPAC] (listening for next command)\n", flush=True)

        deadline   = time.time() + FOLLOWUP_TIMEOUT
        buf        = []
        collecting = False
        silent     = 0.0

        while self._running and time.time() < deadline:
            try:
                data, _ = stream.read(CHUNK)
                energy  = float(abs(data).mean())
            except Exception:
                time.sleep(0.05)
                continue

            # While TTS speaking, watch for interrupt only
            if _tts_speaking.is_set():
                if energy > INTERRUPT_THRESHOLD:
                    self._interrupt_and_listen(stream)
                    deadline = time.time() + FOLLOWUP_TIMEOUT
                buf, collecting, silent = [], False, 0.0
                continue

            if energy > MIC_ENERGY_THRESHOLD:
                collecting = True
                silent     = 0.0
                deadline   = time.time() + FOLLOWUP_TIMEOUT
                buf.append(data.copy())
            elif collecting:
                buf.append(data.copy())
                silent += CHUNK / RATE
                if silent >= 2.0:
                    audio = np.concatenate(buf, axis=0)
                    cmd   = _transcribe(self.stt._model, audio).strip()
                    if cmd.strip():
                        print(f"\n  [OPAC] You said: {cmd}", flush=True)
                        logger.info(f"Follow-up command: '{cmd}'")
                        self.callback(cmd)
                        deadline = time.time() + FOLLOWUP_TIMEOUT
                    buf, collecting, silent = [], False, 0.0

        print(f"\n  [OPAC] (back to wake word mode)\n", flush=True)

    def _listen_for_command(self, stream, already_collected=None) -> str:
        """Listen for a command on the open stream with live display."""
        if not self.stt or not self.stt._model:
            return ""

        print("  [OPAC] You said: ", end="", flush=True)

        full_buf   = list(already_collected) if already_collected else []
        cmd_buf    = []
        collecting = False
        silent     = 0.0
        partials   = []
        deadline   = time.time() + 10.0

        if already_collected:
            audio   = np.concatenate(already_collected, axis=0)
            partial = _transcribe(self.stt._model, audio).strip()
            if partial:
                partials.append(partial)
                print(partial + " ", end="", flush=True)
            deadline = time.time() + 5.0

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

                if silent >= 0.6:
                    burst   = np.concatenate(cmd_buf, axis=0)
                    partial = _transcribe(self.stt._model, burst).strip()
                    if partial:
                        partials.append(partial)
                        print(partial + " ", end="", flush=True)
                    cmd_buf    = []
                    collecting = False
                    silent     = 0.0

                if silent >= 2.5:
                    break

        print(flush=True)

        if not full_buf:
            return ""

        full_audio = np.concatenate(full_buf, axis=0)
        final      = _transcribe(self.stt._model, full_audio).strip()

        if final:
            live = " ".join(partials)
            if final.lower().strip() != live.lower().strip():
                print(f"  [OPAC] (heard) {final}", flush=True)
            logger.info(f"Command: '{final}'")

        return final