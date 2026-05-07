"""
Shared TTS state.
- speaking: set while OPAC TTS is playing audio
- interrupt_requested: set when human wants to interrupt
"""
import threading

speaking           = threading.Event()  # True while TTS plays
interrupt_requested = threading.Event() # True when human interrupts