"""
OPAC -- Objective-driven Planning and Action Coordinator
=========================================================
Run with:
    python opac.py                   # text mode (Phase 1+2+3.5)
    python opac.py --voice           # voice mode (Phase 3)
    python opac.py --voice-only      # fully hands-free voice mode
    python opac.py --file FILE       # summarize a file
    python opac.py --url URL         # summarize a URL
    python opac.py --search TOPIC    # search Wikipedia
    python opac.py --setup           # download and compile model
    python opac.py --info            # show system status
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import OPACAgent
from utils.logger import get_logger
from utils.platform_info import print_system_info

logger = get_logger("opac.main")

BANNER = r"""
  ___  ____   _    ____
 / _ \|  _ \ / \  / ___|
| | | | |_) / _ \| |
| |_| |  __/ ___ \ |___
 \___/|_| /_/   \_\____|

 Objective-driven Planning and Action Coordinator
 Phase 1+2+3+3.5  |  Intel NPU  |  Offline  |  Private
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="OPAC -- Objective-driven Planning and Action Coordinator"
    )
    p.add_argument("--setup",      action="store_true", help="Download + compile model")
    p.add_argument("--file",       type=str,            help="Summarize a file")
    p.add_argument("--url",        type=str,            help="Summarize a URL")
    p.add_argument("--search",     type=str,            help="Search Wikipedia")
    p.add_argument("--voice",      action="store_true", help="Enable voice input+output")
    p.add_argument("--voice-only", action="store_true", help="Fully hands-free voice mode")
    p.add_argument("--info",       action="store_true", help="Print system info")
    p.add_argument("--model",      type=str,            help="Override model name")
    p.add_argument("--device",     type=str, default=None, help="Override device (NPU/GPU/CPU)")
    return p.parse_args()


def main():
    args = parse_args()
    print(BANNER)

    if args.info:
        print_system_info()
        return

    agent = OPACAgent(
        model_override  = args.model,
        device_override = args.device,
    )

    if args.setup:
        agent.setup()
        return

    if not agent.is_ready():
        print("\n[OPAC] Model not found. Run: python opac.py --setup\n")
        sys.exit(1)

    # One-shot modes
    if args.file:
        agent.start()
        result = agent.summarize_file(args.file)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60 + "\n")
        return

    if args.url:
        agent.start()
        result = agent.summarize_url(args.url)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60 + "\n")
        return

    if args.search:
        agent.start()
        agent._init_wiki()
        agent._do_wiki_search(args.search)
        return

    # Voice-only mode (no keyboard)
    if getattr(args, "voice_only", False):
        agent.run_voice_mode()
        return

    # Interactive mode (with optional voice)
    agent.run_interactive(voice=args.voice)


if __name__ == "__main__":
    main()
