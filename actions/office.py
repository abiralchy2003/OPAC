"""
OPAC Office Engine  (Phase 5.5)
==================================
Session-based voice control for Word, PowerPoint, Excel, Browser, Messaging.
"""

from __future__ import annotations

import os, re, platform, subprocess, time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from utils.logger import get_logger

logger     = get_logger("opac.office")
IS_WINDOWS = platform.system() == "Windows"

FOLDER_ALIASES = {
    "downloads": Path.home() / "Downloads",
    "desktop":   Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "pictures":  Path.home() / "Pictures",
    "home":      Path.home(),
}

def _resolve_path(name: str, folder: str, ext: str) -> Path:
    base = FOLDER_ALIASES.get(folder.lower().strip(), Path.home() / "Downloads")
    base.mkdir(parents=True, exist_ok=True)
    fn = re.sub(r'[<>:"/\\|?*]', '', name).strip() or "opac_document"
    if not fn.lower().endswith(ext):
        fn += ext
    return base / fn

def _open_file(path: Path):
    try:
        if IS_WINDOWS:
            os.startfile(str(path))
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        logger.error(f"Cannot open: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WORD SESSION
# ══════════════════════════════════════════════════════════════════════════════
class WordSession:
    def __init__(self):
        try:
            from docx import Document
        except ImportError:
            raise ImportError("pip install python-docx")
        from docx import Document
        self._doc  = Document()
        self._path = None
        # Remove default empty paragraph
        for p in list(self._doc.paragraphs):
            if not p.text.strip():
                p._element.getparent().remove(p._element)
                break

    @property
    def active(self): return self._doc is not None

    def add_heading(self, text: str, level: int = 1) -> str:
        self._doc.add_heading(text, level=level)
        return f"{'Heading' if level==1 else 'Subheading'} added: {text}"

    def add_paragraph(self, text: str) -> str:
        self._doc.add_paragraph(text)
        return f"Paragraph added ({len(text.split())} words)"

    def add_bullets(self, items: List[str]) -> str:
        for item in items:
            self._doc.add_paragraph(item.strip(), style="List Bullet")
        return f"Added {len(items)} bullet points"

    def add_numbered(self, items: List[str]) -> str:
        for item in items:
            self._doc.add_paragraph(item.strip(), style="List Number")
        return f"Added {len(items)} numbered items"

    def add_table(self, rows: int, cols: int, headers: List[str] = None) -> str:
        t = self._doc.add_table(rows=rows, cols=cols)
        t.style = "Table Grid"
        if headers:
            for i, h in enumerate(headers[:cols]):
                t.rows[0].cells[i].text = h
        return f"Added {rows}x{cols} table"

    def add_page_break(self) -> str:
        self._doc.add_page_break()
        return "Page break added"

    def save(self, name: str, folder: str) -> Tuple[bool, str]:
        path = _resolve_path(name, folder, ".docx")
        try:
            self._doc.save(str(path))
            self._path = path
            _open_file(path)
            return True, f"Saved: {path.name} in {path.parent.name}"
        except Exception as e:
            return False, f"Save failed: {e}"

    def summary(self) -> str:
        n = len([p for p in self._doc.paragraphs if p.text.strip()])
        return f"Word: {n} paragraphs"


# ══════════════════════════════════════════════════════════════════════════════
# POWERPOINT SESSION
# ══════════════════════════════════════════════════════════════════════════════
class PowerPointSession:
    LAYOUT_TITLE   = 0
    LAYOUT_CONTENT = 1
    LAYOUT_SECTION = 2
    LAYOUT_BLANK   = 6

    def __init__(self, title: str = ""):
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("pip install python-pptx")
        from pptx import Presentation
        self._prs   = Presentation()
        self._path  = None
        self._count = 0
        if title:
            self.add_title_slide(title)

    @property
    def active(self): return self._prs is not None

    def add_title_slide(self, title: str, subtitle: str = "") -> str:
        layout = self._prs.slide_layouts[self.LAYOUT_TITLE]
        slide  = self._prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        try:
            if subtitle:
                slide.placeholders[1].text = subtitle
        except Exception:
            pass
        self._count += 1
        return f"Title slide: {title}"

    def add_content_slide(self, title: str, content: str) -> str:
        layout = self._prs.slide_layouts[self.LAYOUT_CONTENT]
        slide  = self._prs.slides.add_slide(layout)
        slide.shapes.title.text    = title
        slide.placeholders[1].text = content
        self._count += 1
        return f"Slide {self._count}: {title}"

    def add_bullet_slide(self, title: str, bullets: List[str]) -> str:
        layout = self._prs.slide_layouts[self.LAYOUT_CONTENT]
        slide  = self._prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        tf = slide.placeholders[1].text_frame
        tf.text = bullets[0] if bullets else ""
        for b in bullets[1:]:
            p      = tf.add_paragraph()
            p.text = b
            p.level = 0
        self._count += 1
        return f"Bullet slide {self._count}: {title}"

    def add_section_slide(self, title: str) -> str:
        try:
            layout = self._prs.slide_layouts[self.LAYOUT_SECTION]
        except Exception:
            layout = self._prs.slide_layouts[self.LAYOUT_TITLE]
        slide = self._prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        self._count += 1
        return f"Section slide {self._count}: {title}"

    def add_blank_slide(self) -> str:
        try:
            layout = self._prs.slide_layouts[self.LAYOUT_BLANK]
        except Exception:
            layout = self._prs.slide_layouts[self.LAYOUT_CONTENT]
        self._prs.slides.add_slide(layout)
        self._count += 1
        return f"Blank slide {self._count} added"

    def save(self, name: str, folder: str) -> Tuple[bool, str]:
        path = _resolve_path(name, folder, ".pptx")
        try:
            self._prs.save(str(path))
            self._path = path
            _open_file(path)
            return True, f"Saved: {path.name} in {path.parent.name}"
        except Exception as e:
            return False, f"Save failed: {e}"

    def summary(self) -> str:
        return f"PowerPoint: {self._count} slides"


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL SESSION
# ══════════════════════════════════════════════════════════════════════════════
class ExcelSession:
    def __init__(self):
        try:
            from openpyxl import Workbook
        except ImportError:
            raise ImportError("pip install openpyxl")
        from openpyxl import Workbook
        self._wb       = Workbook()
        self._ws       = self._wb.active
        self._ws.title = "Sheet1"
        self._path     = None
        self._row      = 1

    @property
    def active(self): return self._wb is not None

    def add_headers(self, headers: List[str]) -> str:
        from openpyxl.styles import Font, PatternFill, Alignment
        for c, h in enumerate(headers, 1):
            cell = self._ws.cell(row=self._row, column=c, value=h)
            cell.font      = Font(bold=True, color="FFFFFF")
            cell.fill      = PatternFill("solid", fgColor="4472C4")
            cell.alignment = Alignment(horizontal="center")
        self._row += 1
        return f"Headers: {', '.join(headers)}"

    def add_row(self, values: list) -> str:
        for c, v in enumerate(values, 1):
            self._ws.cell(row=self._row, column=c, value=v)
        self._row += 1
        return f"Row {self._row-1} added"

    def add_total_row(self) -> str:
        from openpyxl.styles import Font
        if self._row < 3:
            return "Not enough data"
        cols = self._ws.max_column
        self._ws.cell(row=self._row, column=1, value="TOTAL").font = Font(bold=True)
        for c in range(2, cols + 1):
            cl  = self._ws.cell(row=1, column=c).column_letter
            f   = f"=SUM({cl}2:{cl}{self._row-1})"
            cell = self._ws.cell(row=self._row, column=c, value=f)
            cell.font = Font(bold=True)
        self._row += 1
        return "Total row added"

    def create_sheet(self, name: str) -> str:
        ws = self._wb.create_sheet(title=name)
        self._ws  = ws
        self._row = 1
        return f"Sheet '{name}' created"

    def _autofit(self):
        for col in self._ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            self._ws.column_dimensions[col[0].column_letter].width = min(w + 4, 40)

    def save(self, name: str, folder: str) -> Tuple[bool, str]:
        self._autofit()
        path = _resolve_path(name, folder, ".xlsx")
        try:
            self._wb.save(str(path))
            self._path = path
            _open_file(path)
            return True, f"Saved: {path.name} in {path.parent.name}"
        except Exception as e:
            return False, f"Save failed: {e}"

    def summary(self) -> str:
        return f"Excel: {self._row-1} rows, {self._ws.max_column} cols, {len(self._wb.sheetnames)} sheet(s)"


# ══════════════════════════════════════════════════════════════════════════════
# BROWSER SESSION
# ══════════════════════════════════════════════════════════════════════════════
class BrowserSession:
    EXE_PATHS = {
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ],
        "brave": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        ],
    }

    def __init__(self):
        self._browser  = None
        self._context  = None
        self._page     = None
        self._pw       = None
        self._name     = ""

    @property
    def active(self): return self._page is not None

    def open(self, name: str = "chrome", profile: str = "") -> Tuple[bool, str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("pip install playwright && playwright install chromium")

        name = name.lower().strip()
        exe  = next((p for p in self.EXE_PATHS.get(name, []) if Path(p).exists()), None)
        args = []

        if profile:
            pdir = self._find_profile(name, profile)
            if pdir:
                args.append(f"--profile-directory={pdir}")

        try:
            self._pw = sync_playwright().start()
            kw = dict(headless=False, args=args)
            if exe:
                kw["executable_path"] = exe
            self._browser = self._pw.chromium.launch(**kw)
            self._context = self._browser.new_context()
            self._page    = self._context.new_page()
            self._page.goto("about:blank")
            self._name    = name
            return True, f"Opened {name.title()}"
        except Exception as e:
            return False, f"Cannot open {name}: {e}"

    def navigate(self, url: str) -> Tuple[bool, str]:
        if not url.startswith("http"):
            url = "https://" + url
        self._page.goto(url, timeout=20000)
        return True, f"Navigated to {self._page.title()}"

    def search_google(self, q: str) -> Tuple[bool, str]:
        self._page.goto(f"https://www.google.com/search?q={q.replace(' ','+')}", timeout=15000)
        return True, f"Searched Google: {q}"

    def search_youtube(self, q: str) -> Tuple[bool, str]:
        self._page.goto(
            f"https://www.youtube.com/results?search_query={q.replace(' ','+')}",
            timeout=15000
        )
        return True, f"Searched YouTube: {q}"

    def new_tab(self, url: str = "about:blank") -> Tuple[bool, str]:
        self._page = self._context.new_page()
        if url != "about:blank":
            self._page.goto(url, timeout=15000)
        return True, "New tab opened"

    def get_page_text(self) -> str:
        try:
            return self._page.evaluate("""() => {
                ['script','style','nav','footer','header','aside']
                    .forEach(t=>document.querySelectorAll(t).forEach(e=>e.remove()));
                return document.body.innerText || '';
            }""")[:3000]
        except Exception:
            return ""

    def whatsapp_send(self, contact: str, message: str) -> Tuple[bool, str]:
        self._page.goto("https://web.whatsapp.com", timeout=25000)
        try:
            self._page.wait_for_selector('[data-testid="chat-list"]', timeout=30000)
            s = self._page.query_selector('[data-testid="search"]')
            if s:
                s.click(); s.type(contact, delay=50); time.sleep(1.5)
                r = self._page.query_selector('[data-testid="cell-frame-container"]')
                if r:
                    r.click(); time.sleep(1)
                    b = self._page.query_selector('[data-testid="conversation-compose-box-input"]')
                    if b:
                        b.click(); b.type(message, delay=30); b.press("Enter")
                        return True, f"WhatsApp message sent to {contact}"
            return False, "Contact not found on WhatsApp Web"
        except Exception as e:
            return False, f"WhatsApp error: {e}"

    def close(self) -> str:
        try:
            if self._browser: self._browser.close()
            if self._pw:      self._pw.stop()
        except Exception:
            pass
        self._browser = self._context = self._page = self._pw = None
        return "Browser closed"

    def _find_profile(self, browser: str, name: str) -> Optional[str]:
        udirs = {
            "chrome": Path.home() / "AppData/Local/Google/Chrome/User Data",
            "edge":   Path.home() / "AppData/Local/Microsoft/Edge/User Data",
            "brave":  Path.home() / "AppData/Local/BraveSoftware/Brave-Browser/User Data",
        }
        udir = udirs.get(browser)
        if not udir or not udir.exists():
            return None
        nl = name.lower()
        if nl in ("default", "1", "first", "main"):
            return "Default"
        m = re.search(r"\d+", name)
        if m:
            n = int(m.group())
            return "Default" if n == 1 else f"Profile {n-1}"
        try:
            import json
            for d in udir.iterdir():
                if d.is_dir() and (d.name == "Default" or d.name.startswith("Profile")):
                    prefs = d / "Preferences"
                    if prefs.exists():
                        pname = json.loads(prefs.read_text(errors="ignore")
                                           ).get("profile", {}).get("name", "").lower()
                        if nl in pname:
                            return d.name
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGING (Desktop apps via pyautogui)
# ══════════════════════════════════════════════════════════════════════════════
class MessagingEngine:
    def send_desktop(self, app: str, contact: str, message: str) -> Tuple[bool, str]:
        try:
            import pyautogui, pyperclip
        except ImportError:
            raise ImportError("pip install pyautogui pyperclip")

        WINDOW_TITLES = {
            "viber":    ["Viber"],
            "telegram": ["Telegram"],
            "discord":  ["Discord"],
            "skype":    ["Skype"],
        }
        titles = WINDOW_TITLES.get(app.lower(), [app.title()])

        if IS_WINDOWS:
            import ctypes
            u32 = ctypes.windll.user32
            hwnd = None
            for t in titles:
                hwnd = u32.FindWindowW(None, t)
                if hwnd:
                    u32.ShowWindow(hwnd, 9)
                    u32.SetForegroundWindow(hwnd)
                    time.sleep(0.8)
                    break
            if not hwnd:
                return False, f"{app} window not found. Make sure {app} is open."

        pyautogui.hotkey("ctrl", "f")
        time.sleep(0.8)
        pyautogui.typewrite(contact, interval=0.05)
        time.sleep(1.2)
        pyautogui.press("enter")
        time.sleep(1.0)

        w, h = pyautogui.size()
        pyautogui.click(w // 2, int(h * 0.92))
        time.sleep(0.5)

        try:
            pyperclip.copy(message)
            pyautogui.hotkey("ctrl", "v")
        except Exception:
            pyautogui.typewrite(message, interval=0.04)

        time.sleep(0.3)
        pyautogui.press("enter")
        return True, f"Message sent to {contact} via {app}"


# ══════════════════════════════════════════════════════════════════════════════
# OFFICE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class OfficeManager:
    """Coordinates all sessions. The agent talks only to this."""

    def __init__(self):
        self.word:    Optional[WordSession]       = None
        self.pptx:    Optional[PowerPointSession] = None
        self.excel:   Optional[ExcelSession]      = None
        self.browser: Optional[BrowserSession]    = None
        self.msg      = MessagingEngine()

    # ── Word ──────────────────────────────────────────────────────────────────

    def start_word(self) -> str:
        self.word = WordSession()
        return ("Word document ready. Commands: 'write about X', "
                "'add heading X', 'add subheading X', 'add bullet list about X', "
                "'add table N columns M rows', 'add page break', "
                "'save as NAME in downloads/desktop/documents'")

    def word_cmd(self, cmd: str, content: str, extra: dict = None) -> str:
        if not self.word:
            return "No Word document open. Say 'open word' first."
        extra = extra or {}
        w = self.word
        if cmd == "heading":       return w.add_heading(content, extra.get("level", 1))
        if cmd == "subheading":    return w.add_heading(content, level=2)
        if cmd == "paragraph":     return w.add_paragraph(content)
        if cmd == "bullets":
            items = [l.strip() for l in content.split("\n") if l.strip()]
            return w.add_bullets(items)
        if cmd == "numbered":
            items = [l.strip() for l in content.split("\n") if l.strip()]
            return w.add_numbered(items)
        if cmd == "table":
            return w.add_table(extra.get("rows", 3), extra.get("cols", 3),
                               extra.get("headers", []))
        if cmd == "page_break":    return w.add_page_break()
        if cmd == "save":
            ok, msg = w.save(extra.get("name","document"), extra.get("folder","downloads"))
            if ok: self.word = None
            return msg
        if cmd == "close":
            self.word = None
            return "Word session closed."
        if cmd == "status":        return w.summary()
        return f"Unknown Word command: {cmd}"

    # ── PowerPoint ────────────────────────────────────────────────────────────

    def start_pptx(self, title: str = "") -> str:
        self.pptx = PowerPointSession(title=title)
        return (f"PowerPoint started: '{title}'. Commands: "
                "'create next slide about X', 'create bullet slide about X', "
                "'create section slide called X', 'create blank slide', "
                "'save as NAME in FOLDER'")

    def pptx_cmd(self, cmd: str, content: str, extra: dict = None) -> str:
        if not self.pptx:
            return "No presentation open. Say 'open powerpoint' first."
        extra = extra or {}
        p = self.pptx
        if cmd == "content_slide":
            return p.add_content_slide(extra.get("title","Slide"), content)
        if cmd == "bullet_slide":
            items = [l.strip() for l in content.split("\n") if l.strip()]
            return p.add_bullet_slide(extra.get("title","Slide"), items)
        if cmd == "title_slide":
            return p.add_title_slide(content, extra.get("subtitle",""))
        if cmd == "section_slide":
            return p.add_section_slide(content)
        if cmd == "blank_slide":
            return p.add_blank_slide()
        if cmd == "save":
            ok, msg = p.save(extra.get("name","presentation"), extra.get("folder","downloads"))
            if ok: self.pptx = None
            return msg
        if cmd == "close":
            self.pptx = None
            return "PowerPoint session closed."
        if cmd == "status":
            return p.summary()
        return f"Unknown PowerPoint command: {cmd}"

    # ── Excel ─────────────────────────────────────────────────────────────────

    def start_excel(self) -> str:
        self.excel = ExcelSession()
        return ("Excel workbook ready. Commands: "
                "'add headers X Y Z', 'add row X Y Z', 'add total row', "
                "'create sheet called NAME', 'save as NAME in FOLDER'")

    def excel_cmd(self, cmd: str, content: str, extra: dict = None) -> str:
        if not self.excel:
            return "No Excel workbook open. Say 'open excel' first."
        extra = extra or {}
        e = self.excel
        if cmd == "headers":
            h = [x.strip() for x in re.split(r"[,\s]+", content) if x.strip()]
            return e.add_headers(h)
        if cmd == "row":
            return e.add_row(_parse_row(content))
        if cmd == "total":
            return e.add_total_row()
        if cmd == "sheet":
            return e.create_sheet(content)
        if cmd == "save":
            ok, msg = e.save(extra.get("name","spreadsheet"), extra.get("folder","downloads"))
            if ok: self.excel = None
            return msg
        if cmd == "close":
            self.excel = None
            return "Excel session closed."
        if cmd == "status":
            return e.summary()
        return f"Unknown Excel command: {cmd}"

    # ── Browser ───────────────────────────────────────────────────────────────

    def start_browser(self, name: str = "chrome", profile: str = "") -> Tuple[bool, str]:
        self.browser = BrowserSession()
        return self.browser.open(name, profile)

    def browser_cmd(self, cmd: str, content: str = "") -> Tuple[bool, str]:
        if not self.browser or not self.browser.active:
            return False, "No browser open. Say 'open chrome/brave/edge' first."
        b = self.browser
        if cmd == "navigate":      return b.navigate(content)
        if cmd == "search":        return b.search_google(content)
        if cmd == "search_youtube":return b.search_youtube(content)
        if cmd == "new_tab":       return b.new_tab(content or "about:blank")
        if cmd == "read":
            t = b.get_page_text()
            return True, t if t else "Page is empty"
        if cmd == "whatsapp":
            parts   = content.split("|", 1)
            contact = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else ""
            return b.whatsapp_send(contact, message)
        if cmd == "close":
            msg = b.close()
            self.browser = None
            return True, msg
        return False, f"Unknown browser command: {cmd}"

    # ── Messaging ─────────────────────────────────────────────────────────────

    def send_message(self, app: str, contact: str, message: str) -> Tuple[bool, str]:
        if app.lower() == "whatsapp" and self.browser and self.browser.active:
            return self.browser.whatsapp_send(contact, message)
        return self.msg.send_desktop(app, contact, message)

    # ── Status / close all ────────────────────────────────────────────────────

    def status(self) -> str:
        parts = []
        if self.word:    parts.append(self.word.summary())
        if self.pptx:    parts.append(self.pptx.summary())
        if self.excel:   parts.append(self.excel.summary())
        if self.browser and self.browser.active: parts.append("Browser: open")
        return "\n  ".join(parts) if parts else "No active sessions."

    def close_all(self):
        self.word = self.pptx = self.excel = None
        if self.browser:
            try: self.browser.close()
            except Exception: pass
        self.browser = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_row(text: str) -> list:
    parts = re.split(r"[,\t]+|\s{2,}", text.strip())
    result = []
    for p in parts:
        p = p.strip()
        if not p: continue
        try:    result.append(int(p))
        except ValueError:
            try:    result.append(float(p))
            except: result.append(p)
    return result


def parse_save(text: str) -> Tuple[str, str]:
    m = re.search(r"save\s+(?:as\s+)?(.+?)\s+(?:in|to|into|at)\s+(\w+)\s*$", text, re.I)
    if m: return m.group(1).strip(), m.group(2).lower()
    m = re.search(r"save\s+(?:as\s+)?(.+)$", text, re.I)
    if m: return m.group(1).strip(), "downloads"
    return "document", "downloads"


def parse_slide(text: str) -> Tuple[str, str]:
    m = re.search(r"title\s+(.+?)\s+and\s+content\s+(.+)$", text, re.I)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(?:about|on|covering|for)\s+(.+)$", text, re.I)
    if m:
        t = m.group(1).strip()
        return t.title(), t
    m = re.search(r"called\s+(.+)$", text, re.I)
    if m:
        t = m.group(1).strip()
        return t, t
    return "Slide", text
