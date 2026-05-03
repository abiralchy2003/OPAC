"""
OPAC App Launcher  (Phase 5)
==============================
Opens any installed application by name — works with both typed
commands and voice ("open Chrome", "launch VS Code", "start Spotify").

How it finds apps:
  Windows:
    1. Start Menu search (searches AppData, ProgramData, Desktop)
    2. Windows Registry (HKLM/HKCU App Paths)
    3. PATH executables
    4. Microsoft Store apps (via explorer shell:AppsFolder)
    5. Common known app aliases (chrome, vscode, notepad, etc.)

  Linux:
    1. .desktop files (/usr/share/applications, ~/.local/share/applications)
    2. PATH executables
    3. Common known app aliases

Usage:
    launcher = AppLauncher()
    success, message = launcher.open("chrome")
    success, message = launcher.open("visual studio code")
    success, message = launcher.list_apps()
"""

from __future__ import annotations

import difflib
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from utils.logger import get_logger

logger = get_logger("opac.launcher")

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# ── Known app aliases ──────────────────────────────────────────────────────────
# Maps common spoken names → executable name or command
WINDOWS_ALIASES: Dict[str, str] = {
    # Browsers
    "chrome":                "chrome.exe",
    "google chrome":         "chrome.exe",
    "edge":                  "msedge.exe",
    "microsoft edge":        "msedge.exe",
    "firefox":               "firefox.exe",
    "brave":                 "brave.exe",
    "opera":                 "opera.exe",
    # Editors / Dev
    "vscode":                "code.exe",
    "vs code":               "code.exe",
    "visual studio code":    "code.exe",
    "visual studio":         "devenv.exe",
    "notepad":               "notepad.exe",
    "notepad++":             "notepad++.exe",
    "sublime":               "sublime_text.exe",
    "sublime text":          "sublime_text.exe",
    "cursor":                "cursor.exe",
    "pycharm":               "pycharm64.exe",
    "intellij":              "idea64.exe",
    "android studio":        "studio64.exe",
    "git bash":              "git-bash.exe",
    # System
    "calculator":            "calc.exe",
    "paint":                 "mspaint.exe",
    "file explorer":         "explorer.exe",
    "explorer":              "explorer.exe",
    "task manager":          "taskmgr.exe",
    "control panel":         "control.exe",
    "cmd":                   "cmd.exe",
    "command prompt":        "cmd.exe",
    "powershell":            "powershell.exe",
    "terminal":              "wt.exe",
    "windows terminal":      "wt.exe",
    "settings":              "ms-settings:",
    "word":                  "winword.exe",
    "excel":                 "excel.exe",
    "powerpoint":            "powerpnt.exe",
    "outlook":               "outlook.exe",
    "teams":                 "teams.exe",
    "microsoft teams":       "teams.exe",
    "zoom":                  "zoom.exe",
    "discord":               "discord.exe",
    "slack":                 "slack.exe",
    "spotify":               "spotify.exe",
    "vlc":                   "vlc.exe",
    "steam":                 "steam.exe",
    "obs":                   "obs64.exe",
    "snipping tool":         "snippingtool.exe",
    "paint 3d":              "mspaint.exe",
    "photos":                "ms-photos:",
    "clock":                 "ms-clock:",
    "weather":               "ms-msn-weather:",
    "store":                 "ms-windows-store:",
    "skype":                 "skype.exe",
    "whatsapp":              "whatsapp.exe",
    "telegram":              "telegram.exe",
    "postman":               "postman.exe",
    "docker":                "docker desktop.exe",
    "wsl":                   "wsl.exe",
}

LINUX_ALIASES: Dict[str, str] = {
    "chrome":                "google-chrome",
    "google chrome":         "google-chrome",
    "firefox":               "firefox",
    "brave":                 "brave-browser",
    "vscode":                "code",
    "vs code":               "code",
    "visual studio code":    "code",
    "gedit":                 "gedit",
    "nautilus":              "nautilus",
    "file manager":          "nautilus",
    "calculator":            "gnome-calculator",
    "terminal":              "gnome-terminal",
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
        self._app_cache: Dict[str, str] = {}   # name -> path/command
        self._cache_built = False

    def open(self, app_name: str) -> Tuple[bool, str]:
        """
        Find and launch an application by name.
        Returns (success, message).
        """
        name = app_name.strip().lower()
        logger.info(f"App launcher: requested '{name}'")

        # Build cache on first use
        if not self._cache_built:
            self._build_cache()

        # Try exact alias first
        cmd = self._resolve_alias(name)
        if cmd:
            return self._launch(cmd, app_name)

        # Try cache (from Start Menu / .desktop files)
        match = self._find_in_cache(name)
        if match:
            return self._launch(match, app_name)

        # Try fuzzy match in cache
        fuzzy = self._fuzzy_find(name)
        if fuzzy:
            return self._launch(fuzzy, app_name)

        return False, (
            f"Could not find '{app_name}' on your system.\n"
            f"  Try: 'open chrome', 'open notepad', 'open vscode'\n"
            f"  Or use the full path: open C:\\Program Files\\...\\app.exe"
        )

    def list_apps(self, filter_str: str = "") -> Tuple[bool, str]:
        """Return a formatted list of launchable apps."""
        if not self._cache_built:
            self._build_cache()

        names = sorted(self._app_cache.keys())
        if filter_str:
            names = [n for n in names if filter_str.lower() in n.lower()]

        if not names:
            return False, "No apps found."

        # Format as columns
        lines = ["  Launchable apps:\n"]
        col_width = max(len(n) for n in names) + 4
        cols = max(1, 70 // col_width)
        row = []
        for n in names:
            row.append(n.ljust(col_width))
            if len(row) >= cols:
                lines.append("  " + "".join(row))
                row = []
        if row:
            lines.append("  " + "".join(row))
        return True, "\n".join(lines)

    # ── cache building ────────────────────────────────────────────────────────

    def _build_cache(self):
        """Scan the system for installed applications."""
        self._app_cache = {}
        if IS_WINDOWS:
            self._scan_windows()
        elif IS_LINUX:
            self._scan_linux()
        self._cache_built = True
        logger.info(f"App cache built: {len(self._app_cache)} apps found")

    def _scan_windows(self):
        """Scan Windows for installed apps."""
        # 1. Scan Start Menu folders
        start_menu_dirs = [
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("ProgramData", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path.home() / "Desktop",
        ]
        for d in start_menu_dirs:
            if d.exists():
                for f in d.rglob("*.lnk"):
                    name = f.stem.lower()
                    self._app_cache[name] = str(f)
                for f in d.rglob("*.exe"):
                    name = f.stem.lower()
                    self._app_cache[name] = str(f)

        # 2. Scan registry App Paths
        try:
            import winreg
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    key = winreg.OpenKey(hive, key_path)
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey = winreg.OpenKey(key, subkey_name)
                            try:
                                val, _ = winreg.QueryValueEx(subkey, "")
                                name = subkey_name.replace(".exe", "").lower()
                                self._app_cache[name] = val
                            except Exception:
                                pass
                            i += 1
                        except OSError:
                            break
                except Exception:
                    pass
        except ImportError:
            pass

        # 3. Scan PATH
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for d in path_dirs:
            try:
                p = Path(d)
                if p.exists():
                    for f in p.glob("*.exe"):
                        self._app_cache[f.stem.lower()] = str(f)
            except Exception:
                pass

    def _scan_linux(self):
        """Scan Linux .desktop files and PATH."""
        desktop_dirs = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path.home() / ".local/share/applications",
            Path("/var/lib/snapd/desktop/applications"),
            Path("/var/lib/flatpak/exports/share/applications"),
        ]
        for d in desktop_dirs:
            if d.exists():
                for f in d.glob("*.desktop"):
                    try:
                        content = f.read_text(errors="ignore")
                        name_match = re.search(r"^Name=(.+)$", content, re.M)
                        exec_match = re.search(r"^Exec=(.+?)(?:\s+%[uUfF])?$", content, re.M)
                        if name_match and exec_match:
                            name = name_match.group(1).strip().lower()
                            cmd  = exec_match.group(1).strip()
                            self._app_cache[name] = cmd
                    except Exception:
                        pass

        # PATH executables
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for d in path_dirs:
            try:
                p = Path(d)
                if p.exists():
                    for f in p.iterdir():
                        if f.is_file() and os.access(f, os.X_OK):
                            self._app_cache[f.name.lower()] = str(f)
            except Exception:
                pass

    # ── resolution ────────────────────────────────────────────────────────────

    def _resolve_alias(self, name: str) -> Optional[str]:
        """Check known aliases first."""
        aliases = WINDOWS_ALIASES if IS_WINDOWS else LINUX_ALIASES
        if name in aliases:
            return aliases[name]
        # Partial match in aliases
        for alias, cmd in aliases.items():
            if name in alias or alias in name:
                return cmd
        return None

    def _find_in_cache(self, name: str) -> Optional[str]:
        """Exact or substring match in app cache."""
        if name in self._app_cache:
            return self._app_cache[name]
        for app_name, path in self._app_cache.items():
            if name in app_name:
                return path
        return None

    def _fuzzy_find(self, name: str) -> Optional[str]:
        """Fuzzy match app name against cache."""
        names = list(self._app_cache.keys())
        matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            logger.info(f"Fuzzy app match: '{name}' -> '{best}'")
            return self._app_cache[best]
        return None

    # ── launch ────────────────────────────────────────────────────────────────

    def _launch(self, cmd: str, display_name: str) -> Tuple[bool, str]:
        """Actually launch the app."""
        try:
            if IS_WINDOWS:
                # ms-settings: and similar URI schemes use explorer
                if cmd.startswith("ms-"):
                    subprocess.Popen(
                        ["explorer.exe", cmd],
                        creationflags=subprocess.DETACHED_PROCESS
                    )
                elif cmd.endswith(".lnk"):
                    subprocess.Popen(
                        ["cmd", "/c", "start", "", cmd],
                        creationflags=subprocess.DETACHED_PROCESS,
                        shell=True
                    )
                else:
                    subprocess.Popen(
                        cmd,
                        creationflags=subprocess.DETACHED_PROCESS,
                        shell=True
                    )
            else:
                # Linux — strip %u %f etc and run detached
                cmd_clean = re.sub(r"%[a-zA-Z]", "", cmd).strip()
                subprocess.Popen(
                    cmd_clean,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )

            msg = f"Opening {display_name} ..."
            logger.info(f"Launched: {cmd}")
            return True, msg

        except Exception as e:
            logger.error(f"Launch failed for '{cmd}': {e}")
            return False, f"Failed to open {display_name}: {e}"
