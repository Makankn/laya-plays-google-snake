from __future__ import annotations

import ctypes
import sys

from .model import Direction


VIRTUAL_KEYS = {
    Direction.LEFT: 0x25,
    Direction.UP: 0x26,
    Direction.RIGHT: 0x27,
    Direction.DOWN: 0x28,
}
KEYEVENTF_KEYUP = 0x0002


class WindowsKeyboard:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("keyboard injection is currently implemented only for Windows")

    def press(self, direction: Direction) -> None:
        key = VIRTUAL_KEYS[direction]
        user32 = ctypes.windll.user32
        user32.keybd_event(key, 0, 0, 0)
        user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
