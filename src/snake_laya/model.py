from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, order=True)
class Cell:
    x: int
    y: int

    def adjacent(self, other: "Cell") -> bool:
        return abs(self.x - other.x) + abs(self.y - other.y) == 1

    def as_list(self) -> list[int]:
        return [self.x, self.y]


class Direction(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @classmethod
    def between(cls, previous: Cell, current: Cell) -> "Direction":
        delta = (current.x - previous.x, current.y - previous.y)
        directions = {
            (0, -1): cls.UP,
            (0, 1): cls.DOWN,
            (-1, 0): cls.LEFT,
            (1, 0): cls.RIGHT,
        }
        if delta not in directions:
            raise ValueError(f"cells are not orthogonally adjacent: {previous} -> {current}")
        return directions[delta]


@dataclass(frozen=True)
class BoardGeometry:
    x: int
    y: int
    width_px: int
    height_px: int
    cell_size_px: int
    columns: int
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameObservation:
    geometry: BoardGeometry
    snake_cells: frozenset[Cell]
    food: Cell | None
    confidence: float
    head_hint: Cell | None = None
    head_center_px: tuple[float, float] | None = None
    head_confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "board_size": [self.geometry.columns, self.geometry.rows],
            "board_roi_px": {
                "x": self.geometry.x,
                "y": self.geometry.y,
                "width": self.geometry.width_px,
                "height": self.geometry.height_px,
            },
            "cell_size_px": self.geometry.cell_size_px,
            "snake_cells": [cell.as_list() for cell in sorted(self.snake_cells, key=lambda c: (c.y, c.x))],
            "head_hint": None if self.head_hint is None else self.head_hint.as_list(),
            "head_center_px": None
            if self.head_center_px is None
            else [round(value, 2) for value in self.head_center_px],
            "head_confidence": round(self.head_confidence, 4),
            "food": None if self.food is None else self.food.as_list(),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class SnakeState:
    geometry: BoardGeometry
    snake: tuple[Cell, ...]  # head first, tail last
    food: Cell | None
    direction: Direction
    frame_index: int

    @property
    def head(self) -> Cell:
        return self.snake[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "board_size": [self.geometry.columns, self.geometry.rows],
            "snake": [cell.as_list() for cell in self.snake],
            "head": self.head.as_list(),
            "food": None if self.food is None else self.food.as_list(),
            "direction": self.direction.value,
            "frame_index": self.frame_index,
        }
