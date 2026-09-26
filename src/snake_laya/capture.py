from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mss
import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture:
    def __init__(self, region: CaptureRegion | None = None, monitor: int = 1) -> None:
        self._region = region
        self._monitor = monitor
        self._mss: Any = None

    def __enter__(self) -> "ScreenCapture":
        self._mss = mss.mss()
        if self._region is None and not 0 <= self._monitor < len(self._mss.monitors):
            raise ValueError(f"monitor {self._monitor} does not exist")
        return self

    def __exit__(self, *args: object) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    def grab(self) -> np.ndarray:
        if self._mss is None:
            raise RuntimeError("ScreenCapture must be used as a context manager")
        target = self._region.as_mss() if self._region is not None else self._mss.monitors[self._monitor]
        bgra = np.asarray(self._mss.grab(target))
        return bgra[:, :, 2::-1].copy()
