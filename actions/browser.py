"""
OPAC Browser Engine  (Phase 4)
================================
Grabs content from the current browser tab and summarises it.

Supports:
  - Chrome / Edge / Brave / Firefox (via Playwright)
  - Fallback: reads URL from clipboard if browser control fails

Install:
    pip install playwright
    playwright install chromium

Usage:
    browser = BrowserEngine()
    url, text = browser.get_current_tab()
    # then pass text to summarizer

Commands added to OPAC:
    "summarize tab"          -- grab current browser tab and summarise
    "summarize browser"      -- same
    "what is this page"      -- same
    "read this page"         -- same
    "summarize current tab"  -- same
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger("opac.browser")


class BrowserEngine:
    def __init__(self):
        self._playwright_ok = None   # None = not checked yet

    def get_current_tab(self) -> Tuple[str, str]:
        """
        Get URL and text content of the current browser tab.
        Returns (url, text) tuple.
        Raises RuntimeError if no browser tab can be found.
        """
        # Try Playwright first (most reliable)
        result = self._try_playwright()
        if result:
            return result

        # Try reading from clipboard (user copies URL manually)
        result = self._try_clipboard_url()
        if result:
            return result

        raise RuntimeError(
            "Cannot access browser tab.\n"
            "  Option 1: Install Playwright:  pip install playwright && playwright install chromium\n"
            "  Option 2: Copy the page URL and paste it into OPAC"
        )

    def get_tab_by_url(self, url: str) -> Tuple[str, str]:
        """Fetch a specific URL and return (url, text)."""
        return self._fetch_url(url)

    def is_available(self) -> bool:
        """Check if Playwright is installed."""
        if self._playwright_ok is None:
            try:
                import playwright
                self._playwright_ok = True
            except ImportError:
                self._playwright_ok = False
        return self._playwright_ok

    # ── Playwright method ─────────────────────────────────────────────────────

    def _try_playwright(self) -> Optional[Tuple[str, str]]:
        """
        Use Playwright to connect to a running browser and get
        the current tab's URL and page text.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.debug("Playwright not installed")
            return None

        try:
            with sync_playwright() as p:
                # Try to connect to Chrome/Edge on default debug port
                # Browser must be started with --remote-debugging-port=9222
                try:
                    browser = p.chromium.connect_over_cdp("http://localhost:9222")
                    context = browser.contexts[0]
                    page    = context.pages[0]
                    url     = page.url
                    # Extract readable text — strips scripts, styles, nav
                    text    = page.evaluate("""() => {
                        const remove = ['script','style','nav','footer',
                                        'header','aside','noscript'];
                        remove.forEach(tag => {
                            document.querySelectorAll(tag).forEach(el => el.remove());
                        });
                        return document.body.innerText || document.body.textContent || '';
                    }""")
                    text = _clean_page_text(text)
                    logger.info(f"Browser tab captured via CDP: {url} ({len(text)} chars)")
                    return (url, text)
                except Exception as e:
                    logger.debug(f"CDP connect failed: {e}")

                # Try launching a new browser and navigating (fallback)
                return None

        except Exception as e:
            logger.debug(f"Playwright error: {e}")
            return None

    # ── Clipboard URL fallback ────────────────────────────────────────────────

    def _try_clipboard_url(self) -> Optional[Tuple[str, str]]:
        """
        Check if the clipboard contains a URL, then fetch it.
        """
        try:
            url = self._get_clipboard()
            if url and re.match(r"https?://\S+", url.strip()):
                url = url.strip()
                logger.info(f"Using URL from clipboard: {url}")
                return self._fetch_url(url)
        except Exception as e:
            logger.debug(f"Clipboard read failed: {e}")
        return None

    def _get_clipboard(self) -> str:
        """Read text from clipboard cross-platform."""
        import platform
        system = platform.system()
        try:
            if system == "Windows":
                import subprocess
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=3
                )
                return result.stdout.strip()
            elif system == "Linux":
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                result = subprocess.run(
                    ["xsel", "--clipboard", "--output"],
                    capture_output=True, text=True, timeout=3
                )
                return result.stdout.strip()
            elif system == "Darwin":
                result = subprocess.run(
                    ["pbpaste"],
                    capture_output=True, text=True, timeout=3
                )
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    # ── URL fetch ─────────────────────────────────────────────────────────────

    def _fetch_url(self, url: str) -> Tuple[str, str]:
        """Fetch a URL and extract readable text."""
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "noscript"]):
                tag.decompose()

            # Try to find main content
            main = (soup.find("main") or soup.find("article") or
                    soup.find(id="content") or soup.find(class_="content") or
                    soup.find("body"))
            text = main.get_text(separator="\n") if main else soup.get_text()
            text = _clean_page_text(text)
            logger.info(f"Fetched URL: {url} ({len(text)} chars)")
            return (url, text)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")


def _clean_page_text(text: str) -> str:
    """Clean raw page text for LLM consumption."""
    import re
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove lines that are just punctuation or single chars
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    return "\n".join(lines).strip()
