from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

from .capture import CaptureRegion


EDGE_MARGIN = 18
MINIMUM_SIZE = 140


def select_region(
    output: str | Path = "capture-region.json",
    *,
    initial_width: int = 650,
    initial_height: int = 580,
    initial_region: CaptureRegion | None = None,
) -> CaptureRegion | None:
    """Show a draggable/resizable translucent rectangle and save its bounds."""
    if sys.platform != "win32":
        raise RuntimeError("the interactive region selector currently supports Windows")
    _enable_per_monitor_dpi()

    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.34)
    root.configure(background="#00E5FF")

    if initial_region is not None:
        left, top = initial_region.left, initial_region.top
        initial_width, initial_height = initial_region.width, initial_region.height
    else:
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        left = max(0, (screen_width - initial_width) // 2)
        top = max(0, (screen_height - initial_height) // 2)
    root.geometry(f"{initial_width}x{initial_height}+{left}+{top}")

    canvas = tk.Canvas(root, background="#00E5FF", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    drag: dict[str, int | str] = {}
    selected: list[CaptureRegion] = []

    def redraw(_event: object | None = None) -> None:
        width = root.winfo_width()
        height = root.winfo_height()
        canvas.delete("all")
        canvas.create_rectangle(4, 4, width - 5, height - 5, outline="#001A21", width=8)
        handle = 12
        for x, y in (
            (4, 4),
            (width // 2, 4),
            (width - 5, 4),
            (4, height // 2),
            (width - 5, height // 2),
            (4, height - 5),
            (width // 2, height - 5),
            (width - 5, height - 5),
        ):
            canvas.create_rectangle(
                x - handle,
                y - handle,
                x + handle,
                y + handle,
                fill="#001A21",
                outline="#FFFFFF",
                width=2,
            )
        canvas.create_rectangle(20, 20, min(width - 20, 620), 74, fill="#001A21", outline="")
        canvas.create_text(
            32,
            47,
            anchor="w",
            fill="white",
            font=("Segoe UI", 13, "bold"),
            text="DRAG TO MOVE  •  RESIZE EDGES  •  ENTER TO SAVE  •  ESC TO CANCEL",
        )

    def edge_mode(x: int, y: int) -> str:
        width, height = root.winfo_width(), root.winfo_height()
        horizontal = "w" if x <= EDGE_MARGIN else "e" if x >= width - EDGE_MARGIN else ""
        vertical = "n" if y <= EDGE_MARGIN else "s" if y >= height - EDGE_MARGIN else ""
        return vertical + horizontal or "move"

    def press(event: tk.Event) -> None:
        drag.update(
            mode=edge_mode(event.x, event.y),
            pointer_x=event.x_root,
            pointer_y=event.y_root,
            left=root.winfo_x(),
            top=root.winfo_y(),
            width=root.winfo_width(),
            height=root.winfo_height(),
        )

    def motion(event: tk.Event) -> None:
        if not drag:
            return
        dx = event.x_root - int(drag["pointer_x"])
        dy = event.y_root - int(drag["pointer_y"])
        mode = str(drag["mode"])
        x, y = int(drag["left"]), int(drag["top"])
        width, height = int(drag["width"]), int(drag["height"])

        if mode == "move":
            x += dx
            y += dy
        else:
            if "e" in mode:
                width = max(MINIMUM_SIZE, width + dx)
            if "s" in mode:
                height = max(MINIMUM_SIZE, height + dy)
            if "w" in mode:
                new_width = max(MINIMUM_SIZE, width - dx)
                x += width - new_width
                width = new_width
            if "n" in mode:
                new_height = max(MINIMUM_SIZE, height - dy)
                y += height - new_height
                height = new_height
        root.geometry(f"{width}x{height}{x:+d}{y:+d}")

    def release(_event: tk.Event) -> None:
        drag.clear()

    def confirm(_event: object | None = None) -> None:
        root.update_idletasks()
        selected.append(
            CaptureRegion(root.winfo_x(), root.winfo_y(), root.winfo_width(), root.winfo_height())
        )
        root.destroy()

    def cancel(_event: object | None = None) -> None:
        root.destroy()

    canvas.bind("<Configure>", redraw)
    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", motion)
    canvas.bind("<ButtonRelease-1>", release)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", cancel)
    root.after(100, root.focus_force)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()

    if not selected:
        return None
    region = selected[0]
    destination = Path(output)
    destination.write_text(json.dumps(region.as_mss(), indent=2) + "\n", encoding="utf-8")
    return region


def load_region(path: str | Path) -> CaptureRegion:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return CaptureRegion(
            left=int(values["left"]),
            top=int(values["top"]),
            width=int(values["width"]),
            height=int(values["height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid capture-region file: {path}") from error


def _enable_per_monitor_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
