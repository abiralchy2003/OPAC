"""
OPAC — Objective-driven Planning and Action Coordinator
Phases 1+2+3+3.5+4+5

Run modes:
    python opac.py                    # interactive text mode
    python opac.py --voice            # text + voice
    python opac.py --voice-only       # fully hands-free
    python opac.py --file FILE        # summarise a file
    python opac.py --url URL          # summarise a URL
    python opac.py --tab              # summarise current browser tab
    python opac.py --open APP         # open an application
    python opac.py --search TOPIC     # search Wikipedia
    python opac.py --setup            # download model
    python opac.py --info             # system info
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import OPACAgent
from utils.logger import get_logger

logger = get_logger("opac.main")

BANNER = r"""
  ___  ____   _    ____
 / _ \|  _ \ / \  / ___|
| | | | |_) / _ \| |
| |_| |  __/ ___ \ |___
 \___/|_| /_/   \_\____|

 Objective-driven Planning and Action Coordinator
 Phase 1+2+3+3.5+4+5  |  Intel NPU  |  Offline  |  Private
"""


def parse_args():
    p = argparse.ArgumentParser(description="OPAC AI Agent")
    p.add_argument("--setup",       action="store_true", help="Download + compile model")
    p.add_argument("--file",        type=str,            help="Summarise a file")
    p.add_argument("--url",         type=str,            help="Summarise a URL")
    p.add_argument("--tab",         action="store_true", help="Summarise current browser tab")
    p.add_argument("--open",        type=str,            help="Open an application")
    p.add_argument("--search",      type=str,            help="Search Wikipedia")
    p.add_argument("--voice",       action="store_true", help="Enable voice input+output")
    p.add_argument("--voice-only",  action="store_true", help="Fully hands-free voice mode")
    p.add_argument("--info",        action="store_true", help="Print system info")
    p.add_argument("--model",       type=str,            help="Override model name")
    p.add_argument("--device",      type=str, default=None, help="Override device")
    return p.parse_args()


def main():
    args = parse_args()
    print(BANNER)

    if args.info:
        try:
            from utils.platform_info import print_system_info
            print_system_info()
        except Exception:
            print("  Run: python opac.py for system info\n")
        return

    agent = OPACAgent(
        model_override  = args.model,
        device_override = args.device,
    )

    if args.setup:
        agent.setup()
        return

    if not agent.is_ready():
        print("\n  [OPAC] Model not found. Run: python opac.py --setup\n")
        sys.exit(1)

    agent.start()

    # One-shot modes
    if args.file:
        result = agent.summarize_file(args.file)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60 + "\n")
        return

    if args.url:
        result = agent.summarize_url(args.url)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60 + "\n")
        return

    if args.tab:
        agent._summarize_current_tab()
        return

    if args.open:
        result = agent._open_app(args.open)
        print(f"\n  OPAC: {result}\n")
        return

    if args.search:
        agent._init_wiki()
        agent._do_wiki_search(args.search)
        return

    if getattr(args, "voice_only", False):
        agent.run_voice_mode()
        return

    agent.run_interactive(voice=args.voice)


if __name__ == "__main__":
    main()
