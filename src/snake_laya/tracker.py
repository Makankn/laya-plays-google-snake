from __future__ import annotations

from .model import Cell, Direction, FrameObservation, SnakeState


class StateTracker:
    """Build logical state only from confirmed eye-cell observations.

    Blue occupancy is deliberately excluded from head tracking: rounded body
    corners and leading caps can enter neighbouring cells before the true head.
    """

    def __init__(
        self,
        initial_length: int = 3,
        *,
        confirmation_frames: int = 1,
        minimum_head_confidence: float = 0.0,
    ) -> None:
        if initial_length < 2:
            raise ValueError("initial_length must be at least 2")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        self._state: SnakeState | None = None
        self._geometry = None
        self._frame_index = -1
        self._logical_length = initial_length
        self._growth_food: Cell | None = None
        self._replacement_food: Cell | None = None
        self._replacement_seen = False
        self._consumed_food: Cell | None = None
        self._confirmed_head: Cell | None = None
        self._candidate_head: Cell | None = None
        self._candidate_frames = 0
        self._confirmation_frames = confirmation_frames
        self._minimum_head_confidence = minimum_head_confidence
        self._expected_direction: Direction | None = None
        self._rejected_hints = 0
        self._transition_source: str | None = None
        self._desynchronized = False
        self._desync_reason: str | None = None

    @property
    def state(self) -> SnakeState | None:
        return self._state

    @property
    def desynchronized(self) -> bool:
        return self._desynchronized

    @property
    def desync_reason(self) -> str | None:
        return self._desync_reason

    @property
    def rejected_hints(self) -> int:
        return self._rejected_hints

    @property
    def transition_source(self) -> str | None:
        return self._transition_source

    @property
    def minimum_head_confidence(self) -> float:
        return self._minimum_head_confidence

    def mark_desynchronized(self, reason: str = "capture continuity was lost") -> None:
        self._desynchronized = True
        self._desync_reason = reason
        self._state = None

    def expect(self, direction: Direction) -> None:
        # Telemetry only. Sending a key is not proof that the browser accepted
        # it, so screenshots remain authoritative.
        self._expected_direction = direction

    def update(self, observation: FrameObservation) -> SnakeState | None:
        self._frame_index += 1
        if self._desynchronized:
            return None
        geometry = observation.geometry
        if self._geometry is not None and (
            self._geometry.columns != geometry.columns
            or self._geometry.rows != geometry.rows
        ):
            self.mark_desynchronized("board dimensions changed during tracking")
            return None
        self._geometry = geometry
        self._observe_food(observation.food)
        if self._state is not None and self._state.food != self._growth_food:
            self._state = SnakeState(
                geometry=geometry,
                snake=self._state.snake,
                food=self._growth_food,
                direction=self._state.direction,
                frame_index=self._state.frame_index,
            )

        head = observation.head_hint
        if head is None or observation.head_confidence < self._minimum_head_confidence:
            self._candidate_head = None
            self._candidate_frames = 0
            return self._state
        if head == self._candidate_head:
            self._candidate_frames += 1
        else:
            self._candidate_head = head
            self._candidate_frames = 1
        if self._candidate_frames < self._confirmation_frames:
            return self._state

        previous_head = self._confirmed_head
        if previous_head is None:
            self._confirmed_head = head
            return None
        if head == previous_head:
            self._rejected_hints = 0
            return self._state

        targets = (head,)
        transition_source = "confirmed_eyes"
        if not previous_head.adjacent(head):
            dx = head.x - previous_head.x
            dy = head.y - previous_head.y
            straight_distance = abs(dx) + abs(dy) if dx == 0 or dy == 0 else 0
            if straight_distance >= 2:
                step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
                step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
                path = tuple(
                    Cell(previous_head.x + step_x * step, previous_head.y + step_y * step)
                    for step in range(1, straight_distance + 1)
                )
                evidence_present = all(
                    cell in observation.snake_cells for cell in path[:-1]
                )
                if straight_distance == 2 or (
                    straight_distance <= 4 and evidence_present
                ):
                    targets = path
                    transition_source = "confirmed_eyes_interpolated"
                else:
                    self.mark_desynchronized(
                        f"unsupported straight eye jump from {previous_head.as_list()} "
                        f"to {head.as_list()}"
                    )
                    return None
            elif dx != 0 and dy != 0 and abs(dx) + abs(dy) <= 4:
                # At capture time the eyes can skip several cells around a
                # corner. Recover only a shortest, one-turn path whose every
                # missing cell is visibly blue. If both possible L-shaped
                # paths fit the pixels, fail closed instead of guessing.
                paths = _one_turn_paths(previous_head, head)
                visible_paths = tuple(
                    path
                    for path in paths
                    if all(cell in observation.snake_cells for cell in path[:-1])
                    and (
                        self._state is None
                        or Direction.between(previous_head, path[0])
                        != _opposite(self._state.direction)
                    )
                )
                if len(visible_paths) == 1:
                    targets = visible_paths[0]
                    transition_source = (
                        "confirmed_eyes_interpolated_turn"
                        if len(targets) == 2
                        else "confirmed_eyes_interpolated_corner"
                    )
                else:
                    self.mark_desynchronized(
                        f"ambiguous corner eye jump from {previous_head.as_list()} "
                        f"to {head.as_list()}"
                    )
                    return None
            else:
                self.mark_desynchronized(
                    f"confirmed eye head jumped from {previous_head.as_list()} to {head.as_list()}"
                )
                return None

        for target in targets:
            direction = Direction.between(previous_head, target)
            if self._state is None:
                if self._growth_food == target:
                    self._logical_length += 1
                    self._finish_growth(observation.food)
                dx = target.x - previous_head.x
                dy = target.y - previous_head.y
                ordered = tuple(
                    Cell(target.x - dx * index, target.y - dy * index)
                    for index in range(self._logical_length)
                )
                if any(
                    cell.x < 0
                    or cell.y < 0
                    or cell.x >= geometry.columns
                    or cell.y >= geometry.rows
                    for cell in ordered
                ):
                    return None
            else:
                if direction == _opposite(self._state.direction):
                    # Google briefly redraws the face behind the logical head
                    # while eating. A reverse transition is impossible in the
                    # game, so this observation cannot be authoritative. Keep
                    # the last confirmed state and wait for fresh eyes instead
                    # of permanently desynchronizing the run.
                    self._rejected_hints += 1
                    self._candidate_head = None
                    self._candidate_frames = 0
                    return self._state
                eats = self._growth_food == target
                occupied = set(self._state.snake)
                if target in occupied:
                    self.mark_desynchronized(
                        f"confirmed eye head {target.as_list()} overlaps the tracked body"
                    )
                    return None
                if eats:
                    self._logical_length += 1
                    self._finish_growth(observation.food)
                ordered = tuple([target, *self._state.snake][: self._logical_length])

            self._confirmed_head = target
            previous_head = target
            self._state = SnakeState(
                geometry=geometry,
                snake=ordered,
                food=self._growth_food,
                direction=direction,
                frame_index=self._frame_index,
            )

        self._rejected_hints = 0
        self._transition_source = transition_source
        return self._state

    def _observe_food(self, observed: Cell | None) -> None:
        if self._growth_food is None:
            if self._consumed_food is not None:
                if observed is None or observed == self._consumed_food:
                    return
                self._consumed_food = None
            if observed is not None:
                self._growth_food = observed
            return
        if observed == self._growth_food:
            if self._replacement_seen and self._replacement_food is None:
                self._replacement_seen = False
            return
        self._replacement_seen = True
        self._replacement_food = observed

    def _finish_growth(self, observed: Cell | None) -> None:
        consumed = self._growth_food
        replacement = self._replacement_food if self._replacement_seen else None
        if replacement is None and observed is not None and observed != consumed:
            replacement = observed
        self._replacement_food = None
        self._replacement_seen = False
        if replacement is None:
            self._growth_food = None
            self._consumed_food = consumed
        else:
            self._growth_food = replacement
            self._consumed_food = None


def _opposite(direction: Direction) -> Direction:
    return {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }[direction]


def _one_turn_paths(start: Cell, end: Cell) -> tuple[tuple[Cell, ...], ...]:
    """Return the two shortest axis-aligned paths with exactly one corner."""

    step_x = 1 if end.x > start.x else -1
    step_y = 1 if end.y > start.y else -1
    horizontal_then_vertical = tuple(
        [
            *(Cell(x, start.y) for x in range(start.x + step_x, end.x + step_x, step_x)),
            *(Cell(end.x, y) for y in range(start.y + step_y, end.y + step_y, step_y)),
        ]
    )
    vertical_then_horizontal = tuple(
        [
            *(Cell(start.x, y) for y in range(start.y + step_y, end.y + step_y, step_y)),
            *(Cell(x, end.y) for x in range(start.x + step_x, end.x + step_x, step_x)),
        ]
    )
    return horizontal_then_vertical, vertical_then_horizontal
