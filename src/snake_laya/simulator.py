from __future__ import annotations

import random
from dataclasses import dataclass

from .model import BoardGeometry, Cell, Direction, SnakeState
from .planner import VECTORS


@dataclass(frozen=True)
class StepResult:
    state: SnakeState | None
    reward: float
    ate: bool
    dead: bool
    repeated: bool


class SnakeSimulator:
    """Small deterministic Snake environment used to adapt Laya off-screen."""

    def __init__(
        self,
        *,
        columns: int = 17,
        rows: int = 15,
        seed: int = 0,
        max_steps: int = 160,
    ) -> None:
        self.geometry = BoardGeometry(0, 0, columns, rows, 1, columns, rows)
        self.random = random.Random(seed)
        self.max_steps = max_steps
        self.state: SnakeState | None = None
        self.steps = 0
        self.apples = 0
        self._visits: dict[tuple[Cell, Cell | None, Direction], int] = {}

    def reset(self) -> SnakeState:
        center_y = self.geometry.rows // 2
        snake = (Cell(4, center_y), Cell(3, center_y), Cell(2, center_y))
        food = self._spawn_food(set(snake))
        self.state = SnakeState(self.geometry, snake, food, Direction.RIGHT, 0)
        self.steps = 0
        self.apples = 0
        self._visits = {}
        self._record_visit(self.state)
        return self.state

    def step(self, direction: Direction) -> StepResult:
        if self.state is None:
            raise RuntimeError("reset the simulator before stepping")
        state = self.state
        dx, dy = VECTORS[direction]
        target = Cell(state.head.x + dx, state.head.y + dy)
        eats = target == state.food
        # Match Google's live collision behaviour: a tight turn into the
        # current tail can die before the animated tail clears the cell.
        occupied = set(state.snake)
        dead = (
            target.x < 0
            or target.y < 0
            or target.x >= self.geometry.columns
            or target.y >= self.geometry.rows
            or target in occupied
        )
        self.steps += 1
        if dead:
            self.state = None
            return StepResult(None, -10.0, False, True, False)

        old_distance = _distance(state.head, state.food)
        snake = (target, *state.snake) if eats else (target, *state.snake[:-1])
        food = self._spawn_food(set(snake)) if eats else state.food
        next_state = SnakeState(
            self.geometry, snake, food, direction, state.frame_index + 1
        )
        self.state = next_state
        if eats:
            self.apples += 1
        new_distance = _distance(target, food if eats else state.food)
        progress = 0.0 if eats else float(old_distance - new_distance)
        repeated = self._record_visit(next_state) > 1
        reward = (10.0 if eats else 0.12 * progress) - 0.01
        if repeated:
            reward -= 0.35
        if self.steps >= self.max_steps:
            reward -= 2.0
        return StepResult(next_state, reward, eats, False, repeated)

    @property
    def done(self) -> bool:
        return self.state is None or self.steps >= self.max_steps

    def _spawn_food(self, occupied: set[Cell]) -> Cell | None:
        free = [
            Cell(x, y)
            for y in range(self.geometry.rows)
            for x in range(self.geometry.columns)
            if Cell(x, y) not in occupied
        ]
        return self.random.choice(free) if free else None

    def _record_visit(self, state: SnakeState) -> int:
        key = (state.head, state.food, state.direction)
        self._visits[key] = self._visits.get(key, 0) + 1
        return self._visits[key]


def _distance(first: Cell, second: Cell | None) -> int:
    if second is None:
        return 0
    return abs(first.x - second.x) + abs(first.y - second.y)
