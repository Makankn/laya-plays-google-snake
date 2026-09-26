from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from PIL import Image

from .model import BoardGeometry, Cell, FrameObservation


@dataclass(frozen=True)
class ParserConfig:
    board_roi: tuple[int, int, int, int] | None = None  # x, y, width, height
    cell_size_px: int | None = None
    minimum_cell_size_px: int = 12
    maximum_cell_size_px: int = 80
    occupied_fraction: float = 0.075
    sample_inset_fraction: float = 0.08


class ParseError(RuntimeError):
    pass


class SnakeFrameParser:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        self._locked_geometry: BoardGeometry | None = None
        self._locked_image_shape: tuple[int, int] | None = None

    def parse_file(self, path: str | Path) -> FrameObservation:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return self.parse_array(rgb)

    def parse_array(self, rgb: np.ndarray) -> FrameObservation:
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ParseError("expected an RGB image")
        rgb = rgb[:, :, :3].astype(np.int16, copy=False)
        image_shape = (rgb.shape[0], rgb.shape[1])
        if self._locked_geometry is not None and self._locked_image_shape == image_shape:
            geometry = self._locked_geometry
        else:
            geometry = self._geometry(rgb)
        board = rgb[
            geometry.y : geometry.y + geometry.height_px,
            geometry.x : geometry.x + geometry.width_px,
        ]

        snake: set[Cell] = set()
        foods: list[tuple[float, Cell]] = []
        classification_strengths: list[float] = []
        size = geometry.cell_size_px
        inset = max(1, round(size * self.config.sample_inset_fraction))

        for y in range(geometry.rows):
            for x in range(geometry.columns):
                cell = board[y * size : (y + 1) * size, x * size : (x + 1) * size]
                core = cell[inset : size - inset, inset : size - inset]
                red, green, blue = (core[:, :, index] for index in range(3))

                blue_mask = (blue > 135) & (blue > red * 1.35) & (blue > green * 1.18)
                red_mask = (red > 175) & (red > green * 1.45) & (red > blue * 1.45)
                blue_fraction = float(blue_mask.mean())
                red_fraction = float(red_mask.mean())
                strongest = max(blue_fraction, red_fraction)

                if blue_fraction >= self.config.occupied_fraction:
                    snake.add(Cell(x, y))
                    classification_strengths.append(min(1.0, blue_fraction / 0.35))
                elif red_fraction >= self.config.occupied_fraction:
                    foods.append((red_fraction, Cell(x, y)))
                    classification_strengths.append(min(1.0, red_fraction / 0.35))
                elif strongest > self.config.occupied_fraction / 2:
                    classification_strengths.append(strongest / self.config.occupied_fraction)

        if not snake:
            raise ParseError("board detected, but no blue snake cells were found")
        food = max(foods, default=(0.0, None), key=lambda item: item[0])[1]
        geometry_score = self._checkerboard_score(board, size)
        object_score = float(np.mean(classification_strengths)) if classification_strengths else 0.0
        confidence = min(1.0, 0.65 * geometry_score + 0.35 * object_score)
        head_hint, head_center, head_confidence = self._detect_head(
            board, size, geometry.columns, geometry.rows
        )
        observation = FrameObservation(
            geometry, frozenset(snake), food, confidence,
            head_hint, head_center, head_confidence,
        )
        if self.config.board_roi is None:
            self._locked_geometry = geometry
            self._locked_image_shape = image_shape
        return observation

    @staticmethod
    def _detect_head(
        board: np.ndarray, size: int, columns: int, rows: int
    ) -> tuple[Cell | None, tuple[float, float] | None, float]:
        """Locate eye whites that are embedded in the blue head sprite."""
        white = np.all(board > 210, axis=2)
        red, green, blue = (board[:, :, index] for index in range(3))
        blue_pixels = (blue > 135) & (blue > red * 1.35) & (blue > green * 1.18)
        remaining = {tuple(point) for point in np.argwhere(white)}
        components: list[list[tuple[int, int]]] = []
        while remaining:
            seed = remaining.pop()
            component = [seed]
            pending = [seed]
            while pending:
                y, x = pending.pop()
                for neighbour in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        pending.append(neighbour)
                        component.append(neighbour)
            components.append(component)

        # Eye whites are large (roughly 70-100 pixels at a 35px cell), while
        # mouth/glint pixels are single-digit components. Scaling the threshold
        # keeps this usable at smaller browser zoom levels.
        minimum_area = max(4, round(size * size * 0.025))
        candidates: list[tuple[float, list[tuple[int, int]]]] = []
        padding = max(2, round(size * 0.22))
        for component in components:
            if len(component) < minimum_area:
                continue
            ys = [point[0] for point in component]
            xs = [point[1] for point in component]
            y0, y1 = max(0, min(ys) - padding), min(board.shape[0], max(ys) + padding + 1)
            x0, x1 = max(0, min(xs) - padding), min(board.shape[1], max(xs) + padding + 1)
            support = float(blue_pixels[y0:y1, x0:x1].mean())
            if support >= 0.08:
                candidates.append((support, component))
        if not candidates:
            return None, None, 0.0

        candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        selected = [candidates[0]]
        first_center = np.mean(np.asarray(candidates[0][1]), axis=0)
        for candidate in candidates[1:]:
            candidate_center = np.mean(np.asarray(candidate[1]), axis=0)
            if float(np.linalg.norm(candidate_center - first_center)) <= size * 0.9:
                selected.append(candidate)
                break
        centers = [np.mean(np.asarray(component), axis=0) for _, component in selected]
        center_y, center_x = np.mean(np.asarray(centers), axis=0)
        cell = Cell(int(center_x // size), int(center_y // size))
        if 0 <= cell.x < columns and 0 <= cell.y < rows:
            component_score = 1.0 if len(selected) == 2 else 0.62
            support_score = min(1.0, max(item[0] for item in selected) / 0.35)
            confidence = 0.65 * component_score + 0.35 * support_score
            return cell, (float(center_x), float(center_y)), confidence
        return None, None, 0.0

    def _geometry(self, rgb: np.ndarray) -> BoardGeometry:
        if self.config.board_roi is None:
            x, y, width, height = self._detect_board(rgb)
        else:
            x, y, width, height = self.config.board_roi

        if self.config.cell_size_px is None:
            cell_size = self._detect_cell_size(rgb[y : y + height, x : x + width])
        else:
            cell_size = self.config.cell_size_px

        columns = round(width / cell_size)
        rows = round(height / cell_size)
        snapped_width = columns * cell_size
        snapped_height = rows * cell_size
        if abs(snapped_width - width) > 2 or abs(snapped_height - height) > 2:
            raise ParseError(
                f"board dimensions {width}x{height} are not multiples of cell size {cell_size}"
            )
        return BoardGeometry(x, y, snapped_width, snapped_height, cell_size, columns, rows)

    @staticmethod
    def _detect_board(rgb: np.ndarray) -> tuple[int, int, int, int]:
        red, green, blue = (rgb[:, :, index] for index in range(3))
        light_green = (
            (green >= 165)
            & (red >= 125)
            & (red <= 205)
            & (blue <= 135)
            & ((green - red) >= 20)
            & ((green - blue) >= 65)
        )
        row_counts = light_green.sum(axis=1)
        # Full-monitor capture is supported, so the board may occupy only a
        # fraction of the screenshot width. The following column-density check
        # still rejects narrow green UI elements.
        candidate_rows = row_counts >= max(80, int(rgb.shape[1] * 0.15))
        y0, y1 = _longest_true_run(candidate_rows)
        if y1 - y0 < 100:
            raise ParseError("could not locate the light-green game board")

        column_counts = light_green[y0:y1].sum(axis=0)
        candidate_columns = column_counts >= int((y1 - y0) * 0.75)
        x0, x1 = _longest_true_run(candidate_columns)
        if x1 - x0 < 100:
            raise ParseError("could not determine board columns")
        return x0, y0, x1 - x0, y1 - y0

    def _detect_cell_size(self, board: np.ndarray) -> int:
        height, width = board.shape[:2]
        common = gcd(width, height)
        candidates = [
            value
            for value in range(self.config.minimum_cell_size_px, self.config.maximum_cell_size_px + 1)
            if width % value == 0 and height % value == 0
        ]
        if common in candidates and width // common >= 6 and height // common >= 6:
            candidates.append(common)
        if not candidates:
            raise ParseError(f"could not find a cell size that divides {width}x{height}")

        scored = [(self._checkerboard_score(board, size), size) for size in set(candidates)]
        score, size = max(scored)
        # Google draws animated sprites across cell boundaries. Their pixels
        # legitimately add variance to both checkerboard parity groups. A real
        # late-game frame in the regression corpus scores 0.240; false divisors
        # remain near zero and still fail this deliberately modest floor.
        if score < 0.20:
            raise ParseError(f"board found, but checkerboard confidence is too low ({score:.3f})")
        return size

    @staticmethod
    def _checkerboard_score(board: np.ndarray, size: int) -> float:
        height, width = board.shape[:2]
        rows, columns = height // size, width // size
        if rows < 4 or columns < 4 or height % size or width % size:
            return 0.0
        inset = max(2, size // 4)
        samples = np.empty((rows, columns, 3), dtype=np.float64)
        for y in range(rows):
            for x in range(columns):
                core = board[
                    y * size + inset : (y + 1) * size - inset,
                    x * size + inset : (x + 1) * size - inset,
                ]
                samples[y, x] = np.median(core.reshape(-1, 3), axis=0)

        parity = np.indices((rows, columns)).sum(axis=0) % 2
        group_a = samples[parity == 0]
        group_b = samples[parity == 1]
        separation = float(np.linalg.norm(group_a.mean(axis=0) - group_b.mean(axis=0)))
        spread = float(group_a.std(axis=0).mean() + group_b.std(axis=0).mean())
        return max(0.0, min(1.0, separation / (separation + spread + 1e-6)))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[::2], edges[1::2]
    if len(starts) == 0:
        return 0, 0
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])
