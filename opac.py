"""
OPAC — Objective-driven Planning and Action Coordinator
========================================================
Main entry point. Run with:
    python opac.py                 # interactive text mode
    python opac.py --voice         # voice mode (Phase 3)
    python opac.py --file FILE     # summarize a file directly
    python opac.py --url URL       # summarize a URL directly
    python opac.py --setup         # download and compile model for NPU
"""

import argparse
import sys
from pathlib import Path

# Make sure our package is on the path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent import OPACAgent
from utils.logger import get_logger
from utils.platform_info import print_system_info

logger = get_logger("opac.main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OPAC — Objective-driven Planning and Action Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python opac.py                          Interactive mode
  python opac.py --setup                  Download + compile model for NPU
  python opac.py --file report.pdf        Summarize a PDF
  python opac.py --file slides.pptx       Summarize a PowerPoint
  python opac.py --url https://example.com Summarize a webpage
  python opac.py --info                   Show system / NPU status
        """
    )
    parser.add_argument("--setup",  action="store_true", help="Download and compile model for NPU (run once)")
    parser.add_argument("--file",   type=str,            help="Path to a file to summarize")
    parser.add_argument("--url",    type=str,            help="URL to summarize")
    parser.add_argument("--voice",  action="store_true", help="Enable voice mode (Phase 3)")
    parser.add_argument("--info",   action="store_true", help="Print system and NPU information")
    parser.add_argument("--model",  type=str,            help="Override model name (default from config)")
    parser.add_argument("--device", type=str,            default=None,
                        help="Override inference device: NPU, GPU, CPU (default: NPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(r"""
  ___  ____   _    ____
 / _ \|  _ \ / \  / ___|
| | | | |_) / _ \| |
| |_| |  __/ ___ \ |___
 \___/|_| /_/   \_\____|

 Objective-driven Planning and Action Coordinator
 Running fully on Intel NPU | Offline | Private
""")

    if args.info:
        print_system_info()
        return

    agent = OPACAgent(
        model_override=args.model,
        device_override=args.device,
    )

    if args.setup:
        agent.setup()
        return

    if not agent.is_ready():
        print("\n[OPAC] Model not found. Run 'python opac.py --setup' first.\n")
        sys.exit(1)

    # --- one-shot modes ---
    if args.file:
        result = agent.summarize_file(args.file)
        print("\n" + "═" * 60)
        print(result)
        print("═" * 60 + "\n")
        return

    if args.url:
        result = agent.summarize_url(args.url)
        print("\n" + "═" * 60)
        print(result)
        print("═" * 60 + "\n")
        return

    # --- interactive mode ---
    agent.run_interactive()


if __name__ == "__main__":
    main()
