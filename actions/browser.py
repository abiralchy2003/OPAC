"""
OPAC Browser Engine  (Phase 4)
================================
Grabs content from the current browser tab and summarises it.

Supported browsers (CDP remote debugging):
  - Chrome     --remote-debugging-port=9222
  - Edge       --remote-debugging-port=9222
  - Brave      --remote-debugging-port=9222
  - Firefox    (via playwright, limited support)

Fallback: reads URL from clipboard if browser control fails.

Install:
    pip install playwright
    playwright install chromium

Start your browser with remote debugging (do this once, make a shortcut):
    Chrome:  chrome.exe --remote-debugging-port=9222
    Edge:    msedge.exe --remote-debugging-port=9222
    Brave:   brave.exe  --remote-debugging-port=9222
"""

from __future__ import annotations

import re
import subprocess
import platform
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger("opac.browser")

# Brave default install paths on Windows
BRAVE_PATHS_WINDOWS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Users\{user}\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe",
]

CHROME_PATHS_WINDOWS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Users\{user}\AppData\Local\Google\Chrome\Application\chrome.exe",
]

EDGE_PATHS_WINDOWS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# How to start each browser with remote debugging for copy-paste instructions
BROWSER_LAUNCH_TIPS = {
    "brave":  r'"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" --remote-debugging-port=9222',
    "chrome": r'"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222',
    "edge":   r'"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9222',
}


class BrowserEngine:
    def __init__(self):
        self._playwright_ok = None

    def get_current_tab(self) -> Tuple[str, str]:
        """
        Get URL and text content of the current browser tab.
        Tries Chrome/Edge/Brave CDP connection first, then clipboard fallback.
        Returns (url, text).
        """
        result = self._try_playwright()
        if result:
            return result

        result = self._try_clipboard_url()
        if result:
            return result

        raise RuntimeError(
            "Cannot access browser tab. Options:\n\n"
            "  OPTION 1 — Start your browser with remote debugging:\n"
            "    Brave:  brave.exe --remote-debugging-port=9222\n"
            "    Chrome: chrome.exe --remote-debugging-port=9222\n"
            "    Edge:   msedge.exe --remote-debugging-port=9222\n\n"
            "  Then install Playwright:\n"
            "    pip install playwright\n"
            "    playwright install chromium\n\n"
            "  OPTION 2 — Copy the page URL (Ctrl+C in address bar)\n"
            "    then type: summarize tab\n"
            "    OPAC will detect it from your clipboard."
        )

    def get_tab_by_url(self, url: str) -> Tuple[str, str]:
        return self._fetch_url(url)

    def is_available(self) -> bool:
        if self._playwright_ok is None:
            try:
                import playwright
                self._playwright_ok = True
            except ImportError:
                self._playwright_ok = False
        return self._playwright_ok

    def get_browser_launch_command(self, browser: str = "brave") -> str:
        """Return the command to start a browser with remote debugging."""
        return BROWSER_LAUNCH_TIPS.get(browser.lower(), BROWSER_LAUNCH_TIPS["brave"])

    # ── Playwright CDP ────────────────────────────────────────────────────────

    def _try_playwright(self) -> Optional[Tuple[str, str]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.debug("Playwright not installed — pip install playwright")
            return None

        try:
            with sync_playwright() as p:
                # All Chromium-based browsers (Brave, Chrome, Edge) use the
                # same Chrome DevTools Protocol on port 9222
                for port in [9222, 9223, 9224]:
                    try:
                        browser = p.chromium.connect_over_cdp(
                            f"http://localhost:{port}"
                        )
                        context = browser.contexts[0]
                        page    = context.pages[0]
                        url     = page.url

                        # Skip browser internal pages
                        if url.startswith(("chrome://", "edge://", "brave://",
                                           "about:", "chrome-extension://")):
                            logger.debug(f"Skipping internal page: {url}")
                            # Try next page in context
                            for pg in context.pages:
                                if not pg.url.startswith(("chrome://", "edge://",
                                                          "brave://", "about:")):
                                    page = pg
                                    url  = page.url
                                    break
                            else:
                                continue

                        text = page.evaluate("""() => {
                            const remove = ['script','style','nav','footer',
                                            'header','aside','noscript',
                                            '.cookie-banner','.advertisement',
                                            '#cookie-notice','[aria-hidden="true"]'];
                            remove.forEach(sel => {
                                try {
                                    document.querySelectorAll(sel)
                                            .forEach(el => el.remove());
                                } catch(e) {}
                            });
                            return document.body.innerText
                                || document.body.textContent
                                || '';
                        }""")
                        text = _clean_page_text(text)
                        logger.info(
                            f"Browser tab captured via CDP port {port}: "
                            f"{url} ({len(text)} chars)"
                        )
                        return (url, text)

                    except Exception as e:
                        logger.debug(f"CDP port {port} failed: {e}")
                        continue

        except Exception as e:
            logger.debug(f"Playwright error: {e}")

        return None

    # ── Clipboard fallback ────────────────────────────────────────────────────

    def _try_clipboard_url(self) -> Optional[Tuple[str, str]]:
        try:
            url = self._get_clipboard().strip()
            if url and re.match(r"https?://\S+", url):
                logger.info(f"Using URL from clipboard: {url}")
                print(f"  [OPAC] Found URL in clipboard: {url}", flush=True)
                return self._fetch_url(url)
        except Exception as e:
            logger.debug(f"Clipboard read failed: {e}")
        return None

    def _get_clipboard(self) -> str:
        system = platform.system()
        try:
            if system == "Windows":
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=3
                )
                return result.stdout.strip()
            elif system == "Linux":
                for cmd in [["xclip", "-selection", "clipboard", "-o"],
                             ["xsel", "--clipboard", "--output"]]:
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                        if r.returncode == 0:
                            return r.stdout.strip()
                    except Exception:
                        continue
            elif system == "Darwin":
                result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=3)
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    # ── URL fetch ─────────────────────────────────────────────────────────────

    def _fetch_url(self, url: str) -> Tuple[str, str]:
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

            main = (soup.find("main") or soup.find("article") or
                    soup.find(id="content") or soup.find(class_="content") or
                    soup.find("body"))
            text = main.get_text(separator="\n") if main else soup.get_text()
            text = _clean_page_text(text)
            logger.info(f"Fetched: {url} ({len(text)} chars)")
            return (url, text)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")


def _clean_page_text(text: str) -> str:
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    return "\n".join(lines).strip()