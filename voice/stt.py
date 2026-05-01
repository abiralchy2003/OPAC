"""
OPAC Speech-to-Text  (Phase 3)
================================
Converts microphone audio to text using faster-whisper (CPU).

Two modes:
  listen()        — records until silence, then transcribes once
  listen_live()   — transcribes in short rolling chunks and prints
                    each partial result so user sees live feedback

Install:
    pip install faster-whisper sounddevice numpy
"""

from __future__ import annotations

import time
from utils.logger import get_logger
from config.settings import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE,
    MIC_SILENCE_TIMEOUT, MIC_ENERGY_THRESHOLD
)

logger = get_logger("opac.voice.stt")


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

    # ── main listen method ────────────────────────────────────────────────────

    def listen(self, timeout: float = 10.0) -> str:
        """
        Record from microphone with LIVE transcription display.

        Shows every chunk of speech as it is heard, so the user
        can see exactly what OPAC is capturing in real time.
        Waits for MIC_SILENCE_TIMEOUT seconds of silence to finish.
        Returns the full transcription as a single string.
        """
        if not self._loaded:
            raise RuntimeError("Call stt.load() first.")

        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError("pip install sounddevice numpy")

        RATE           = 16000
        CHUNK          = 4096          # ~0.25 s per chunk — good balance
        device         = _get_input_device()
        full_buffer    = []            # accumulates all audio
        chunk_buffer   = []            # current speech burst
        collecting     = False
        silent         = 0.0
        partial_texts  = []            # transcription of each burst

        print("  [OPAC] Speak now (I'll show what I hear) ...", flush=True)
        print("  [OPAC] You said: ", end="", flush=True)

        try:
            stream = sd.InputStream(
                samplerate=RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK,
                device=device,
            )
            stream.start()
        except Exception as e:
            logger.error(f"Cannot open microphone for STT: {e}")
            print(f"\n  [OPAC] Mic error: {e}", flush=True)
            return ""

        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                try:
                    data, _ = stream.read(CHUNK)
                    energy  = float(abs(data).mean())
                except Exception:
                    time.sleep(0.05)
                    continue

                if energy > MIC_ENERGY_THRESHOLD:
                    # Speech detected — start/continue collecting
                    collecting = True
                    silent     = 0.0
                    chunk_buffer.append(data.copy())
                    full_buffer.append(data.copy())

                elif collecting:
                    # Silence after speech
                    chunk_buffer.append(data.copy())
                    full_buffer.append(data.copy())
                    silent += CHUNK / RATE

                    if silent >= 0.6:
                        # 0.6 s silence — transcribe what we have so far
                        # and show it live, then keep listening
                        burst = np.concatenate(chunk_buffer, axis=0)
                        partial = self._transcribe(burst).strip()
                        if partial:
                            partial_texts.append(partial)
                            print(partial + " ", end="", flush=True)
                        chunk_buffer = []
                        silent       = 0.0
                        collecting   = False

                    if silent >= MIC_SILENCE_TIMEOUT:
                        # Full silence timeout — done listening
                        break

        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        print()  # newline after live display

        # Transcribe entire audio for the best final result
        if not full_buffer:
            return ""

        import numpy as np
        full_audio = np.concatenate(full_buffer, axis=0)
        final_text = self._transcribe(full_audio).strip()
        logger.info(f"STT final: '{final_text}'")

        # Show correction if final differs from live display
        if final_text:
            live_combined = " ".join(partial_texts)
            if final_text.lower() != live_combined.lower():
                print(f"  [OPAC] (corrected) You said: {final_text}", flush=True)

        return final_text

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file directly (for testing)."""
        if not self._loaded:
            raise RuntimeError("Call stt.load() first.")
        segments, _ = self._model.transcribe(path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── internal ──────────────────────────────────────────────────────────────

    def _transcribe(self, audio) -> str:
        """Transcribe a numpy int16 audio array."""
        import numpy as np
        audio_f32 = audio.flatten().astype(np.float32) / 32768.0
        # Skip very short clips — Whisper hallucinates on <0.3 s of audio
        if len(audio_f32) < 16000 * 0.3:
            return ""
        segments, _ = self._model.transcribe(audio_f32, beam_size=5, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()