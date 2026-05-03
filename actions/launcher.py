"""
OPAC App Launcher  (Phase 5)
==============================
Opens any installed application by name.

How it finds apps:
  1. Known aliases (chrome, vscode, notepad, etc.)
  2. Windows Registry App Paths (HKLM + HKCU)
  3. Start Menu .lnk shortcuts
  4. PATH executables

How it launches:
  - Full path exe  -> subprocess.Popen directly
  - Bare exe name  -> os.startfile() or ShellExecute (Windows finds it)
  - ms-settings:   -> explorer.exe URI
  - .lnk shortcut  -> ShellExecute (resolves target automatically)
"""

from __future__ import annotations

import difflib
import os
import platform
import re
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Dict, Optional
from utils.logger import get_logger

logger = get_logger("opac.launcher")

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# Known aliases: spoken name -> executable name or full command
WINDOWS_ALIASES: Dict[str, str] = {
    # Browsers
    "chrome":                "chrome",
    "google chrome":         "chrome",
    "edge":                  "msedge",
    "microsoft edge":        "msedge",
    "firefox":               "firefox",
    "brave":                 "brave",
    "brave browser":         "brave",
    "opera":                 "opera",
    # Editors / Dev
    "vscode":                "code",
    "vs code":               "code",
    "visual studio code":    "code",
    "visual studio":         "devenv",
    "notepad":               "notepad",
    "notepad++":             "notepad++",
    "sublime":               "subl",
    "sublime text":          "subl",
    "cursor":                "cursor",
    "pycharm":               "pycharm",
    "intellij":              "idea",
    "android studio":        "studio64",
    "git bash":              "git-bash",
    # System tools
    "calculator":            "calc",
    "paint":                 "mspaint",
    "file explorer":         "explorer",
    "explorer":              "explorer",
    "task manager":          "taskmgr",
    "control panel":         "control",
    "cmd":                   "cmd",
    "command prompt":        "cmd",
    "powershell":            "powershell",
    "terminal":              "wt",
    "windows terminal":      "wt",
    # Office / Productivity
    "word":                  "winword",
    "excel":                 "excel",
    "powerpoint":            "powerpnt",
    "outlook":               "outlook",
    "onenote":               "onenote",
    "teams":                 "teams",
    "microsoft teams":       "teams",
    # Communication
    "zoom":                  "zoom",
    "discord":               "discord",
    "slack":                 "slack",
    "skype":                 "skype",
    "whatsapp":              "whatsapp",
    "telegram":              "telegram",
    # Media / Entertainment
    "spotify":               "spotify",
    "vlc":                   "vlc",
    "steam":                 "steam",
    "obs":                   "obs64",
    # Settings URIs (handled specially)
    "settings":              "ms-settings:",
    "windows settings":      "ms-settings:",
    "store":                 "ms-windows-store:",
    "microsoft store":       "ms-windows-store:",
    "photos":                "ms-photos:",
    "clock":                 "ms-clock:",
    # Other
    "postman":               "postman",
    "docker":                "docker desktop",
    "wsl":                   "wsl",
    "snipping tool":         "snippingtool",
}

LINUX_ALIASES: Dict[str, str] = {
    "chrome":                "google-chrome",
    "google chrome":         "google-chrome",
    "firefox":               "firefox",
    "brave":                 "brave-browser",
    "vscode":                "code",
    "vs code":               "code",
    "visual studio code":    "code",
    "terminal":              "gnome-terminal",
    "calculator":            "gnome-calculator",
    "spotify":               "spotify",
    "vlc":                   "vlc",
    "discord":               "discord",
    "slack":                 "slack",
    "zoom":                  "zoom",
    "steam":                 "steam",
    "obs":                   "obs",
}


class AppLauncher:
    def __init__(self):
        self._registry_paths: Dict[str, str] = {}  # exe_name -> full_path
        self._lnk_paths: Dict[str, str]      = {}  # display_name -> lnk_path
        self._cache_built = False

    def open(self, app_name: str) -> Tuple[bool, str]:
        """Find and launch an application. Returns (success, message)."""
        name = app_name.strip().lower()
        logger.info(f"App launcher: requested '{name}'")

        if not self._cache_built:
            self._build_cache()

        # 1. Check aliases
        exe = self._resolve_alias(name)
        if exe:
            return self._launch(exe, app_name)

        # 2. Check if it is already a full path
        p = Path(app_name)
        if p.exists() and p.suffix.lower() in (".exe", ".lnk", ""):
            return self._launch(str(p), app_name)

        # 3. Check registry paths (full paths from App Paths)
        match = self._find_registry(name)
        if match:
            return self._launch(match, app_name)

        # 4. Check Start Menu shortcuts
        match = self._find_lnk(name)
        if match:
            return self._launch(match, app_name)

        # 5. Check PATH
        found = shutil.which(name) or shutil.which(name + ".exe")
        if found:
            return self._launch(found, app_name)

        # 6. Fuzzy match across everything we know
        fuzzy = self._fuzzy_find(name)
        if fuzzy:
            return self._launch(fuzzy, app_name)

        return False, (
            f"Could not find '{app_name}'.\n"
            f"  Try: 'open chrome', 'open notepad', 'open vs code'\n"
            f"  Tip: type 'list apps' to see all available apps."
        )

    def list_apps(self, filter_str: str = "") -> Tuple[bool, str]:
        if not self._cache_built:
            self._build_cache()

        # Combine aliases + registry + shortcuts
        all_names = set(WINDOWS_ALIASES.keys() if IS_WINDOWS else LINUX_ALIASES.keys())
        all_names.update(self._registry_paths.keys())
        all_names.update(self._lnk_paths.keys())

        names = sorted(n for n in all_names
                       if not filter_str or filter_str.lower() in n.lower())
        if not names:
            return False, "No apps found."

        lines = [f"\n  Found {len(names)} apps:\n"]
        col_w = min(max((len(n) for n in names), default=20) + 3, 35)
        cols  = max(1, 72 // col_w)
        row   = []
        for n in names:
            row.append(n.ljust(col_w))
            if len(row) >= cols:
                lines.append("  " + "".join(row))
                row = []
        if row:
            lines.append("  " + "".join(row))
        lines.append("")
        return True, "\n".join(lines)

    # ── cache building ─────────────────────────────────────────────────────────

    def _build_cache(self):
        if IS_WINDOWS:
            self._scan_registry()
            self._scan_start_menu()
        elif IS_LINUX:
            self._scan_desktop_files()
        self._cache_built = True
        logger.info(
            f"App cache built: {len(self._registry_paths)} registry + "
            f"{len(self._lnk_paths)} shortcuts"
        )

    def _scan_registry(self):
        """Scan HKLM and HKCU App Paths for full exe paths."""
        try:
            import winreg
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    key = winreg.OpenKey(hive, key_path)
                    i   = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            try:
                                subkey = winreg.OpenKey(key, subkey_name)
                                val, _ = winreg.QueryValueEx(subkey, "")
                                # val is the full path to the exe
                                if val and Path(val).exists():
                                    name = subkey_name.lower().replace(".exe", "")
                                    self._registry_paths[name] = val
                            except Exception:
                                pass
                            i += 1
                        except OSError:
                            break
                except Exception:
                    pass
        except ImportError:
            pass

    def _scan_start_menu(self):
        """Scan Start Menu for .lnk shortcuts."""
        dirs = [
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("ProgramData", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path.home() / "Desktop",
        ]
        for d in dirs:
            if not d.exists():
                continue
            for f in d.rglob("*.lnk"):
                name = f.stem.lower()
                self._lnk_paths[name] = str(f)

    def _scan_desktop_files(self):
        """Scan Linux .desktop files."""
        desktop_dirs = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path.home() / ".local/share/applications",
        ]
        for d in desktop_dirs:
            if not d.exists():
                continue
            for f in d.glob("*.desktop"):
                try:
                    content = f.read_text(errors="ignore")
                    nm = re.search(r"^Name=(.+)$", content, re.M)
                    ex = re.search(r"^Exec=(.+?)(\s+%[uUfF])?$", content, re.M)
                    if nm and ex:
                        self._lnk_paths[nm.group(1).strip().lower()] = ex.group(1).strip()
                except Exception:
                    pass

    # ── resolution ────────────────────────────────────────────────────────────

    def _resolve_alias(self, name: str) -> Optional[str]:
        aliases = WINDOWS_ALIASES if IS_WINDOWS else LINUX_ALIASES
        # Exact match
        if name in aliases:
            return aliases[name]
        # Partial — name contains alias or alias contains name
        for alias, cmd in aliases.items():
            if name in alias or alias in name:
                return cmd
        return None

    def _find_registry(self, name: str) -> Optional[str]:
        if name in self._registry_paths:
            return self._registry_paths[name]
        for key, path in self._registry_paths.items():
            if name in key:
                return path
        return None

    def _find_lnk(self, name: str) -> Optional[str]:
        if name in self._lnk_paths:
            return self._lnk_paths[name]
        for key, path in self._lnk_paths.items():
            if name in key:
                return path
        return None

    def _fuzzy_find(self, name: str) -> Optional[str]:
        # Combine all known names
        all_items = {}
        aliases = WINDOWS_ALIASES if IS_WINDOWS else LINUX_ALIASES
        for k, v in aliases.items():
            all_items[k] = v
        for k, v in self._registry_paths.items():
            all_items[k] = v
        for k, v in self._lnk_paths.items():
            all_items[k] = v

        matches = difflib.get_close_matches(name, all_items.keys(), n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            logger.info(f"Fuzzy match: '{name}' -> '{best}'")
            return all_items[best]
        return None

    # ── launch ────────────────────────────────────────────────────────────────

    def _launch(self, cmd: str, display_name: str) -> Tuple[bool, str]:
        """Launch the resolved command or path."""
        try:
            if IS_WINDOWS:
                self._launch_windows(cmd)
            else:
                self._launch_linux(cmd)

            msg = f"Opening {display_name} ..."
            logger.info(f"Launched '{display_name}': {cmd}")
            return True, msg

        except Exception as e:
            logger.error(f"Launch failed '{cmd}': {e}")
            return False, f"Failed to open {display_name}: {e}"

    def _launch_windows(self, cmd: str):
        """
        Launch on Windows using the most reliable method for each case.
        """
        # ms-settings: and similar URI schemes
        if re.match(r"^ms-\w+:", cmd):
            subprocess.Popen(["explorer.exe", cmd])
            return

        # .lnk shortcut — use ShellExecute so Windows resolves the target
        if cmd.lower().endswith(".lnk"):
            os.startfile(cmd)
            return

        # Full path to existing exe
        if Path(cmd).exists():
            subprocess.Popen(
                [cmd],
                creationflags=subprocess.DETACHED_PROCESS |
                              subprocess.CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
            return

        # Bare exe name (e.g. "chrome", "code", "notepad")
        # Try shutil.which to resolve to full path first
        full_path = shutil.which(cmd) or shutil.which(cmd + ".exe")
        if full_path:
            subprocess.Popen(
                [full_path],
                creationflags=subprocess.DETACHED_PROCESS |
                              subprocess.CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
            return

        # Last resort: os.startfile lets Windows search its own app registry
        try:
            os.startfile(cmd)
            return
        except Exception:
            pass

        # Final fallback: ShellExecute via PowerShell
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden",
             "-Command", f"Start-Process '{cmd}'"],
            creationflags=subprocess.DETACHED_PROCESS,
        )

    def _launch_linux(self, cmd: str):
        cmd_clean = re.sub(r"%[a-zA-Z]", "", cmd).strip()
        subprocess.Popen(
            cmd_clean,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )