"""
OPAC Wake Word Detection  (Phase 3)
=====================================
Single-stream architecture — one mic stream for everything.

Key behaviours:
  - Fuzzy matching catches Whisper mishearing "hey opac"
  - After a command is processed, OPAC enters FOLLOW-UP mode:
    listens for the next command for up to 8 seconds without
    needing the wake word again. Only returns to wake word
    detection after 8 seconds of silence.
"""

from __future__ import annotations

import difflib
import threading
import time
import numpy as np
from utils.logger import get_logger
from config.settings import MIC_ENERGY_THRESHOLD

logger = get_logger("opac.voice.wakeword")

# What Whisper transcribes "hey opac" as — only real variants, no false positives
WAKE_TRIGGERS = [
    "hey opac", "hey opec", "hey o-back", "hey o back", "hey oh back",
    "hey oh-back", "hey opak", "hi opac", "hi opec", "hello opac",
    "hello opec", "opac", "opec", "cook", "let's cook", "hello",
]

FUZZY_THRESHOLD      = 0.72   # full-phrase similarity
WORD_FUZZY_THRESHOLD = 0.80   # single-word similarity (stricter)

RATE  = 16000
CHUNK = 1024

# How long to stay in follow-up mode after a command (seconds)
# During this time, anything you say is treated as a command without wake word
FOLLOWUP_TIMEOUT = 8.0


def _is_wake_word(text: str) -> bool:
    t = text.lower().strip().rstrip(".!")

    # Must be short — real commands come AFTER the wake word, not during it
    if len(t.split()) > 5:
        return False

    # Exact substring
    for trigger in WAKE_TRIGGERS:
        if trigger in t:
            logger.info(f"Wake word exact: '{trigger}' in '{t}'")
            return True

    # Fuzzy full-phrase
    for trigger in WAKE_TRIGGERS:
        ratio = difflib.SequenceMatcher(None, t, trigger).ratio()
        if ratio >= FUZZY_THRESHOLD:
            logger.info(f"Wake word fuzzy: '{t}' ~ '{trigger}' ({ratio:.2f})")
            return True

    # Word-level — only for very short utterances (1-3 words)
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
    """Beep + visual indicator."""
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
        self.callback = callback
        self.stt      = stt_engine
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Wake word detector started (follow-up mode enabled)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # kept for API compatibility
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
                            cmd = self._listen_for_command(stream)
                            if cmd.strip():
                                self.callback(cmd)
                                # Follow-up mode
                                self._followup_loop(stream)
                            time.sleep(0.3)
            return True
        except Exception as e:
            logger.debug(f"openwakeword error: {e}")
            return False

    def _run_single_stream(self):
        logger.info("Wake word: single-stream STT (follow-up mode enabled)")

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
                                # ── FOLLOW-UP MODE ─────────────────────────
                                # After OPAC processes the command, listen for
                                # more commands without needing wake word again
                                self._followup_loop(stream)
                            time.sleep(0.3)
                        wake_buf, collecting, silent = [], False, 0.0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _followup_loop(self, stream):
        """
        Stay in listening mode after a command.
        Any speech within FOLLOWUP_TIMEOUT seconds is treated as
        the next command — no wake word needed.
        After FOLLOWUP_TIMEOUT seconds of silence, returns to wake word mode.
        """
        print("\n  [OPAC] (Say your next command, or stay quiet to return to wake word mode)",
              flush=True)

        deadline    = time.time() + FOLLOWUP_TIMEOUT
        buf         = []
        collecting  = False
        silent      = 0.0

        while self._running and time.time() < deadline:
            try:
                data, _ = stream.read(CHUNK)
                energy  = float(abs(data).mean())
            except Exception:
                time.sleep(0.05)
                continue

            if energy > MIC_ENERGY_THRESHOLD:
                collecting = True
                silent     = 0.0
                deadline   = time.time() + FOLLOWUP_TIMEOUT  # reset timer on speech
                buf.append(data.copy())
            elif collecting:
                buf.append(data.copy())
                silent += CHUNK / RATE
                if silent >= 2.0:
                    # Transcribe and send as command
                    audio = np.concatenate(buf, axis=0)
                    cmd   = self._listen_for_command(stream, already_collected=buf)
                    # _listen_for_command will handle its own collection
                    # here we just check if we got something already
                    if not cmd:
                        # Use what we already collected
                        cmd = _transcribe(self.stt._model, audio).strip()
                    if cmd.strip():
                        print(f"\n  [OPAC] You said: {cmd}", flush=True)
                        logger.info(f"Follow-up command: '{cmd}'")
                        self.callback(cmd)
                        # Reset follow-up window
                        deadline   = time.time() + FOLLOWUP_TIMEOUT
                        buf        = []
                        collecting = False
                        silent     = 0.0
                    else:
                        # Empty transcription — stop follow-up
                        break

        print(f"\n  [OPAC] (Back to wake word mode — say 'hey opac')\n",
              flush=True)

    def _listen_for_command(self, stream, already_collected=None) -> str:
        """
        Listen for a command on the already-open stream.
        Shows live partial transcriptions.
        Returns the full transcribed command string.
        """
        if not self.stt or not self.stt._model:
            return ""

        print("  [OPAC] You said: ", end="", flush=True)

        full_buf   = list(already_collected) if already_collected else []
        cmd_buf    = []
        collecting = False
        silent     = 0.0
        partials   = []
        deadline   = time.time() + 10.0

        # If we already have audio, transcribe it first
        if already_collected:
            audio   = np.concatenate(already_collected, axis=0)
            partial = _transcribe(self.stt._model, audio).strip()
            if partial:
                partials.append(partial)
                print(partial + " ", end="", flush=True)
            # Then listen for more
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