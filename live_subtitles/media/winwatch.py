"""The foreground window's title and program, over Win32 (no extra dependencies).

Media mode only reads titles from browsers and video players; a title from a chat app
or an editor is never treated as something playing.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass

BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "opera_gx.exe", "vivaldi.exe",
            "arc.exe", "librewolf.exe", "waterfox.exe", "zen.exe", "floorp.exe", "chromium.exe"}
PLAYERS = {"vlc.exe", "mpc-hc64.exe", "mpc-hc.exe", "mpc-be64.exe", "mpc-be.exe", "mpv.exe", "potplayermini64.exe",
           "potplayermini.exe", "potplayer64.exe", "kmplayer.exe", "stremio.exe", "kodi.exe", "plex.exe",
           "jellyfin media player.exe", "jellyfinmediaplayer.exe", "smplayer.exe", "wmplayer.exe", "video.ui.exe",
           "microsoft.media.player.exe"}


@dataclass(frozen=True)
class Window:
    title: str
    exe: str            # lower-case file name, e.g. "chrome.exe"

    @property
    def is_browser(self) -> bool:
        return self.exe in BROWSERS

    @property
    def is_player(self) -> bool:
        return self.exe in PLAYERS

    @property
    def watchable(self) -> bool:
        return self.is_browser or self.is_player


def foreground() -> Window | None:
    if sys.platform != "win32":
        return None
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    exe = ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)      # PROCESS_QUERY_LIMITED_INFORMATION
    if handle:
        try:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                exe = os.path.basename(path.value).lower()
        finally:
            kernel32.CloseHandle(handle)
    return Window(buf.value, exe)
