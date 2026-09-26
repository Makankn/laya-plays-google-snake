"""Small, independent desktop view of the live controller's JSONL telemetry."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
ARROWS = {"UP": "↑", "RIGHT": "→", "DOWN": "↓", "LEFT": "←"}
COLORS = {"UP": "#72C9F4", "RIGHT": "#8DE1AF", "DOWN": "#F1B77D", "LEFT": "#BEA6EC"}


class JsonlTail:
    """Read only complete new lines; tolerate an in-progress writer and truncation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""

    def read(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.offset = 0
                self.pending = b""
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        return events


@dataclass
class OverlayTelemetry:
    status: str = "CONNECTING"
    direction: str | None = None
    current_direction: str | None = None
    probabilities: dict[str, float] | None = None
    probability_label: str = "AWAITING LAYA"
    latency_ms: float | None = None
    board_confidence: float | None = None
    head_confidence: float | None = None
    input_status: str = "Awaiting first turn"
    detail: str = "Waiting for the board"
    updated_at: float = field(default_factory=time.monotonic)

    def ingest(self, event: dict[str, Any]) -> None:
        status = event.get("status")
        self.updated_at = time.monotonic()
        if status == "decision":
            decision = event.get("decision") or {}
            state = event.get("state") or {}
            perception = event.get("perception") or {}
            reason = decision.get("intervention_reason")
            forced = reason in {"forced_turn_guard", "single_executable_move"}
            probabilities = (
                decision.get("lookahead_probabilities")
                if reason == "laya_food_lookahead"
                else decision.get("probabilities")
            )
            self.probabilities = None if forced else _probabilities(probabilities)
            self.probability_label = (
                "SAFETY-ONLY TURN" if forced else
                "LAYA · LOOKAHEAD" if reason == "laya_food_lookahead" else
                "LAYA · MODEL OUTPUT" if self.probabilities else "NO MODEL OUTPUT"
            )
            self.current_direction = decision.get("executed") or state.get("direction")
            self.direction = decision.get("queued_escape") or self.current_direction
            self.latency_ms = _number(decision.get("inference_ms"))
            self.board_confidence = _number(perception.get("board_confidence"))
            self.head_confidence = _number(perception.get("head_confidence"))
            self.status = "GUARDED" if decision.get("intervened") else "LIVE"
            if decision.get("queued_escape"):
                self.detail = f"Buffered for next cell · moving {self.current_direction or '—'} now"
            elif forced:
                self.detail = "Only safe turn · Laya was not queried"
            elif decision.get("intervened"):
                self.detail = f"Safety override · Laya preferred {decision.get('proposed', 'another turn')}"
            else:
                self.detail = "Laya selected this turn"
        elif status == "watching":
            self.status = "WATCHING"
            self._clear_decision("Waiting for a reliable frame")
        elif status == "board_locked":
            self.status = "WATCHING"
            self.detail = "Board locked · waiting for state"
        elif status == "waiting_for_board":
            self.status = "WAITING"
            self._clear_decision("Waiting for a reliable frame")
        elif status == "turn_requested":
            self.input_status = f"{event.get('direction', '—')} sent · awaiting confirmation"
        elif status == "turn_retried":
            self.input_status = f"{event.get('direction', '—')} retried · attempt {event.get('attempt', '—')}"
        elif status == "turn_acknowledged":
            attempts = event.get("attempts", 1)
            suffix = "first try" if attempts == 1 else f"after {attempts} tries"
            self.input_status = f"{event.get('direction', '—')} confirmed · {suffix}"
        elif status == "turn_cancelled":
            self.input_status = "Turn cancelled · no longer safe"
        elif status == "queued_escape_sent":
            self.input_status = f"{event.get('direction', '—')} buffered for next cell"
        elif status == "desynchronized":
            self.status = "DESYNC"
            self._clear_decision("No current decision")
        elif status == "stopped":
            self.status = "STOPPED"
            self._clear_decision("No current decision")

    def _clear_decision(self, detail: str) -> None:
        self.direction = None
        self.current_direction = None
        self.probabilities = None
        self.probability_label = "AWAITING LAYA"
        self.latency_ms = None
        self.board_confidence = None
        self.head_confidence = None
        self.input_status = "No keypresses while tracking is paused"
        self.detail = detail


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _probabilities(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result = {direction: _number(value.get(direction)) for direction in DIRECTIONS}
    if any(number is None for number in result.values()):
        return None
    return {direction: max(0.0, min(1.0, result[direction])) for direction in DIRECTIONS}


def _rounded(canvas: Any, x1: int, y1: int, x2: int, y2: int, radius: int, **kwargs: Any) -> None:
    canvas.create_polygon(
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        smooth=True, splinesteps=12, **kwargs,
    )


def _noactivate(root: Any) -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    user32.GetParent.argtypes = [ctypes.c_void_p]
    user32.GetParent.restype = ctypes.c_void_p
    user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowPos.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]
    handle = root.winfo_id()
    handle = user32.GetParent(handle) or handle
    get_style = user32.GetWindowLongW
    set_style = user32.SetWindowLongW
    style = get_style(handle, -20)
    set_style(handle, -20, style | 0x08000000 | 0x00000080)
    user32.SetWindowPos(handle, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010 | 0x0020)


def place_away_from_board(
    board: tuple[int, int, int, int],
    screen: tuple[int, int],
    size: tuple[int, int] = (354, 448),
) -> tuple[int, int]:
    """Choose an on-screen position that does not cover the capture rectangle."""
    left, top, board_width, board_height = board
    screen_width, screen_height = screen
    width, height = size
    gap = 16
    choices = (
        (left + board_width + gap, top),
        (left - width - gap, top),
        (left, top - height - gap),
        (left, top + board_height + gap),
        (screen_width - width - 24, 88),
    )

    def overlap(x: int, y: int) -> int:
        return max(0, min(x + width, left + board_width) - max(x, left)) * max(
            0, min(y + height, top + board_height) - max(y, top)
        )

    candidates = [
        (max(0, min(x, screen_width - width)), max(0, min(y, screen_height - height)))
        for x, y in choices
    ]
    return min(candidates, key=lambda point: overlap(*point))


def run_overlay(
    path: Path,
    *,
    x: int | None = None,
    y: int = 88,
    avoid_region: tuple[int, int, int, int] | None = None,
) -> None:
    import tkinter as tk

    if sys.platform == "win32":
        from .region_selector import _enable_per_monitor_dpi

        _enable_per_monitor_dpi()
    width, height = 354, 448
    root = tk.Tk()
    root.withdraw()
    root.title("Laya performance")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 1.0)
    root.configure(bg="#09151D")
    if x is None and avoid_region is not None:
        x, y = place_away_from_board(
            avoid_region, (root.winfo_screenwidth(), root.winfo_screenheight()), (width, height)
        )
    elif x is None:
        x = max(0, root.winfo_screenwidth() - width - 24)
    root.geometry(f"{width}x{height}+{x}+{y}")
    canvas = tk.Canvas(root, width=width, height=height, bg="#09151D", highlightthickness=0)
    canvas.pack()
    telemetry = OverlayTelemetry()
    tail = JsonlTail(path)
    drag_origin: tuple[int, int, int, int] | None = None

    def draw() -> None:
        canvas.delete("all")
        _rounded(canvas, 1, 1, width - 1, height - 1, 18, fill="#10232D", outline="#2D4752", width=1)
        canvas.create_rectangle(20, 18, 23, 48, fill="#8DE1AF", outline="")
        canvas.create_text(33, 15, text="LAYA", fill="#F1F7F8", anchor="nw", font=("Segoe UI Semibold", 17))
        canvas.create_text(34, 41, text="LIVE PERFORMANCE", fill="#93AAB3", anchor="nw", font=("Segoe UI", 8))
        status_color = (
            "#8DE1AF" if telemetry.status == "LIVE" else
            "#F4BC83" if telemetry.status in {"GUARDED", "WAITING", "WATCHING"} else
            "#EF9390"
        )
        _rounded(canvas, 230, 19, 322, 44, 11, fill="#1D3741", outline="")
        canvas.create_oval(241, 28, 248, 35, fill=status_color, outline="")
        canvas.create_text(257, 30, text=telemetry.status, fill=status_color, anchor="w", font=("Segoe UI Semibold", 8))
        canvas.create_text(337, 19, text="×", fill="#A3B9C0", anchor="n", font=("Segoe UI", 17))
        canvas.create_line(19, 61, 335, 61, fill="#2D4752")

        status_copy = {
            "LIVE": ("Tracking the game", "Decisions are updating with each new cell"),
            "GUARDED": ("Safety adjusted a turn", "The chosen turn differs from Laya's first pick"),
            "WAITING": ("Board not visible", "Move data is hidden until tracking recovers"),
            "WATCHING": ("Looking for the board", "Keep the game visible in the capture box"),
            "DESYNC": ("Tracking lost", "Restart the run before sending more keys"),
            "STOPPED": ("Session ended", "Start another run to see live decisions"),
            "CONNECTING": ("Connecting to the run", "Waiting for the first board observation"),
        }
        title, subtitle = status_copy.get(telemetry.status, status_copy["CONNECTING"])
        _rounded(canvas, 17, 72, 337, 110, 10, fill="#1A333E", outline="")
        canvas.create_oval(29, 85, 37, 93, fill=status_color, outline="")
        canvas.create_text(46, 76, text=title, fill="#F0F6F7", anchor="nw", font=("Segoe UI Semibold", 10))
        canvas.create_text(46, 94, text=subtitle, fill="#A7BBC2", anchor="nw", font=("Segoe UI", 8))

        direction = telemetry.direction if telemetry.direction in ARROWS else None
        direction_color = "#8DE1AF" if direction else "#8DA7B0"
        _rounded(canvas, 17, 121, 337, 215, 12, fill="#162F3A", outline="#27434D", width=1)
        canvas.create_text(31, 133, text="NEXT TURN", fill="#A4BAC2", anchor="nw", font=("Segoe UI Semibold", 9))
        canvas.create_text(29, 151, text=ARROWS.get(direction, "·"), fill=direction_color, anchor="nw", font=("Segoe UI Semibold", 34))
        canvas.create_text(78, 159, text=direction or "Paused", fill="#F4F8F8", anchor="nw", font=("Segoe UI Semibold", 21))
        choice_confidence = (
            None if telemetry.probabilities is None or direction is None
            else telemetry.probabilities[direction]
        )
        canvas.create_text(322, 134, text="LAYA SUPPORT", fill="#8DA7B0", anchor="ne", font=("Segoe UI Semibold", 8))
        canvas.create_text(322, 150, text="—" if choice_confidence is None else f"{choice_confidence:.0%}", fill=direction_color, anchor="ne", font=("Segoe UI Semibold", 17))
        canvas.create_text(31, 196, text=telemetry.detail, fill="#AFC4C9", anchor="nw", font=("Segoe UI", 9))

        canvas.create_text(20, 220, text="MOVE PROBABILITIES", fill="#A4BAC2", anchor="nw", font=("Segoe UI Semibold", 9))
        canvas.create_text(333, 220, text=telemetry.probability_label, fill="#819CA7", anchor="ne", font=("Segoe UI", 8))
        preferred = None if telemetry.probabilities is None else max(telemetry.probabilities, key=telemetry.probabilities.get)
        for index, direction_name in enumerate(DIRECTIONS):
            row_y = 246 + index * 26
            probability = None if telemetry.probabilities is None else telemetry.probabilities[direction_name]
            canvas.create_text(21, row_y - 2, text=f"{ARROWS[direction_name]}  {direction_name}", fill="#D8E5E8", anchor="nw", font=("Segoe UI Semibold", 9))
            _rounded(canvas, 106, row_y + 2, 282, row_y + 11, 4, fill="#28434D", outline="")
            if probability is not None and probability > 0:
                bar_color = "#8DE1AF" if direction_name == preferred else "#668B99"
                _rounded(canvas, 106, row_y + 2, 106 + max(8, round(176 * probability)), row_y + 11, 4, fill=bar_color, outline="")
            canvas.create_text(331, row_y - 2, text="—" if probability is None else f"{probability:.0%}", fill="#F0F6F7", anchor="ne", font=("Segoe UI Semibold", 9))

        metrics = (
            ("DECISION TIME", "—" if telemetry.latency_ms is None else f"{telemetry.latency_ms:.0f} ms", 17, 118),
            ("BOARD", "—" if telemetry.board_confidence is None else f"{telemetry.board_confidence:.0%}", 125, 228),
            ("HEAD", "—" if telemetry.head_confidence is None else f"{telemetry.head_confidence:.0%}", 235, 337),
        )
        for label, value, left, right in metrics:
            _rounded(canvas, left, 355, right, 405, 9, fill="#1A333E", outline="")
            canvas.create_text(left + 10, 363, text=label, fill="#91AAB4", anchor="nw", font=("Segoe UI Semibold", 8))
            canvas.create_text(left + 10, 379, text=value, fill="#F0F6F7", anchor="nw", font=("Segoe UI Semibold", 15))

        canvas.create_line(19, 416, 335, 416, fill="#2D4752")
        canvas.create_text(20, 425, text="KEYS", fill="#91AAB4", anchor="nw", font=("Segoe UI Semibold", 8))
        canvas.create_text(62, 425, text=telemetry.input_status[:40], fill="#C7D8DC", anchor="nw", font=("Segoe UI", 8))

    def poll() -> None:
        events = tail.read()
        if events:
            for event in events:
                telemetry.ingest(event)
            draw()
        root.after(120, poll)

    def begin_drag(event: Any) -> None:
        nonlocal drag_origin
        if event.x >= 325 and event.y < 60:
            root.destroy()
            return
        if event.y < 62 and event.x < 325:
            drag_origin = (event.x_root, event.y_root, root.winfo_x(), root.winfo_y())

    def drag(event: Any) -> None:
        if drag_origin is not None:
            sx, sy, ox, oy = drag_origin
            root.geometry(f"+{ox + event.x_root - sx}+{oy + event.y_root - sy}")

    def end_drag(_event: Any) -> None:
        nonlocal drag_origin
        drag_origin = None

    canvas.bind("<ButtonPress-1>", begin_drag)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", end_drag)
    draw()
    root.update_idletasks()
    _noactivate(root)
    root.deiconify()
    _noactivate(root)
    poll()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Floating Laya live-performance overlay")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int, default=88)
    parser.add_argument("--avoid-region", type=lambda value: tuple(int(part) for part in value.split(",")))
    args = parser.parse_args()
    if args.avoid_region is not None and len(args.avoid_region) != 4:
        parser.error("--avoid-region must be LEFT,TOP,WIDTH,HEIGHT")
    run_overlay(args.log, x=args.x, y=args.y, avoid_region=args.avoid_region)


if __name__ == "__main__":
    main()
