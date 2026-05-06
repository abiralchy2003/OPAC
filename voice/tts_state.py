"""
Shared flag: True while OPAC's TTS is actively playing audio.
The wake word detector checks this and ignores mic input during playback
so OPAC cannot interrupt itself with its own speaker output.
"""
import threading

# Set by TTSEngine when speaking starts, cleared when speaking ends
speaking = threading.Event()
