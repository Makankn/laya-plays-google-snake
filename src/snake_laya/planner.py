from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import Cell, Direction, SnakeState


VECTORS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


@dataclass(frozen=True)
class MoveInfo:
    direction: Direction
    target: Cell
    legal: bool
    safe: bool
    eats: bool
    reachable_cells: int
    food_distance: int
    reason: str
    boundary: bool
    corner: bool
    forward_clearance: int
    operationally_safe: bool
    operational_reason: str
    requires_escape: bool


class SafetyPlanner:
    """Conservative planner for Google's odd-by-odd default board.

    The laya-mlx demo uses a Hamiltonian-cycle shield, but a 17x15 grid cannot
    contain a Hamiltonian cycle. Here a move is safe only when the simulated
    head can still reach the simulated tail through unoccupied cells.
    """

    def analyse(self, state: SnakeState) -> tuple[MoveInfo, ...]:
        return tuple(self._analyse_move(state, direction) for direction in Direction)

    def preferred(self, moves: tuple[MoveInfo, ...]) -> Direction | None:
        candidates = [move for move in moves if move.safe and move.operationally_safe]
        if not candidates:
            candidates = [move for move in moves if move.legal and move.operationally_safe]
        if not candidates:
            return None
        # Entering a boundary cell head-on leaves very little time for the
        # screenshot loop to observe the cell and send the escape turn. Travel
        # one row/column inside the wall and enter only to eat an edge apple.
        def rank(move: MoveInfo) -> tuple[int, int, int]:
            boundary_transit = int(move.boundary and not move.eats)
            return boundary_transit, move.food_distance, -move.reachable_cells

        return min(candidates, key=rank).direction

    def _analyse_move(self, state: SnakeState, direction: Direction) -> MoveInfo:
        dx, dy = VECTORS[direction]
        target = Cell(state.head.x + dx, state.head.y + dy)
        columns, rows = state.geometry.columns, state.geometry.rows
        boundary = target.x in (0, columns - 1) or target.y in (0, rows - 1)
        corner = target.x in (0, columns - 1) and target.y in (0, rows - 1)
        forward_clearance = {
            Direction.UP: target.y,
            Direction.DOWN: rows - 1 - target.y,
            Direction.LEFT: target.x,
            Direction.RIGHT: columns - 1 - target.x,
        }[direction]
        eats = target == state.food
        reason = "legal"

        if not (0 <= target.x < columns and 0 <= target.y < rows):
            reason = "wall"
        elif len(state.snake) > 1 and target == state.snake[1]:
            reason = "reverse"
        else:
            # Google's continuously animated body can still collide with the
            # logical tail while that cell is being vacated.
            occupied = set(state.snake)
            if target in occupied:
                reason = "body"

        if reason != "legal":
            return MoveInfo(
                direction, target, False, False, eats, 0, 10**9, reason,
                boundary, corner, forward_clearance, False, reason, False,
            )

        simulated = (target, *state.snake) if eats else (target, *state.snake[:-1])
        blocked = set(simulated[1:-1])
        reachable = _flood(target, blocked, columns, rows)
        tail_reachable = simulated[-1] in reachable
        has_exit = _has_two_step_exit(simulated, columns, rows, straight=eats)
        safe = tail_reachable and has_exit
        reason = (
            "legal" if safe else
            "no legal exit" if not has_exit else
            "no path to tail"
        )
        distance = (
            abs(target.x - state.food.x) + abs(target.y - state.food.y)
            if state.food is not None
            else 0
        )
        # The outer cell and its adjacent inner lane need a preselected escape:
        # Google's apple redraw can precede logical eye entry, leaving less
        # than one inference window to turn after a near-wall bite.
        requires_escape = safe and forward_clearance <= 1
        operationally_safe = safe and not requires_escape
        operational_reason = "reliable"
        if not safe:
            operational_reason = reason
        elif requires_escape:
            operational_reason = "requires a queued escape before the next wall tick"
        return MoveInfo(
            direction, target, True, safe, eats, len(reachable), distance, reason,
            boundary, corner, forward_clearance, operationally_safe, operational_reason,
            requires_escape,
        )


def _has_two_step_exit(
    snake: tuple[Cell, ...], columns: int, rows: int, *, straight: bool
) -> bool:
    if len(snake) == columns * rows:
        return True
    head = snake[0]
    occupied = set(snake)
    for dx, dy in VECTORS.values():
        first = Cell(head.x + dx, head.y + dy)
        if not (
            0 <= first.x < columns
            and 0 <= first.y < rows
            and first not in occupied
        ):
            continue
        next_snake = (first, *snake[:-1])
        next_occupied = set(next_snake)
        steps = ((dx, dy),) if straight else VECTORS.values()
        if any(
            0 <= first.x + step_x < columns
            and 0 <= first.y + step_y < rows
            and Cell(first.x + step_x, first.y + step_y) not in next_occupied
            for step_x, step_y in steps
        ):
            return True
    return False


def _flood(start: Cell, blocked: set[Cell], columns: int, rows: int) -> set[Cell]:
    visited = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for dx, dy in VECTORS.values():
            neighbour = Cell(cell.x + dx, cell.y + dy)
            if (
                0 <= neighbour.x < columns
                and 0 <= neighbour.y < rows
                and neighbour not in blocked
                and neighbour not in visited
            ):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited
