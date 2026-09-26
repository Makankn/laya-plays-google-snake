from snake_laya.controller import LiveController, _PendingEscape, _PendingTurn
from snake_laya.model import BoardGeometry, Cell, Direction, FrameObservation, SnakeState
from snake_laya.policy import Decision
from snake_laya.tracker import StateTracker


class FakeKeyboard:
    def __init__(self) -> None:
        self.pressed: list[Direction] = []

    def press(self, direction: Direction) -> None:
        self.pressed.append(direction)


def test_queued_escape_is_sent_on_edge_apple_redraw() -> None:
    emitted: list[str] = []
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
        emit=emitted.append,
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(15, 12),
        target=Cell(16, 12),
        direction=Direction.UP,
        food_before=Cell(16, 12),
        eats=True,
    )
    observation = FrameObservation(
        geometry=BoardGeometry(0, 0, 595, 525, 35, 17, 15),
        snake_cells=frozenset({Cell(15, 12)}),
        food=Cell(4, 3),
        confidence=1.0,
        head_hint=Cell(15, 12),
        head_center_px=(553.0, 437.5),
        head_confidence=1.0,
    )

    controller._send_pending_escape_if_ready(observation)

    assert keyboard.pressed == [Direction.UP]
    assert controller._pending_escape.sent
    assert any('"trigger": "food_redraw"' in event for event in emitted)


def test_approach_escape_waits_for_edge_apple_redraw() -> None:
    emitted: list[str] = []
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
        emit=emitted.append,
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(5, 12),
        target=Cell(5, 13),
        direction=Direction.RIGHT,
        food_before=Cell(5, 14),
        eats=True,
        trigger_cell=Cell(5, 14),
        trigger_on_target=False,
    )
    before_redraw = FrameObservation(
        geometry=BoardGeometry(0, 0, 595, 525, 35, 17, 15),
        snake_cells=frozenset({Cell(5, 12), Cell(5, 13)}),
        food=Cell(5, 14),
        confidence=1.0,
        head_hint=Cell(5, 13),
        head_center_px=(192.5, 472.5),
        head_confidence=1.0,
    )

    controller._send_pending_escape_if_ready(before_redraw)
    assert keyboard.pressed == []

    after_redraw = FrameObservation(
        geometry=before_redraw.geometry,
        snake_cells=before_redraw.snake_cells,
        food=Cell(9, 3),
        confidence=1.0,
        head_hint=Cell(5, 13),
        head_center_px=(192.5, 472.5),
        head_confidence=1.0,
    )
    controller._send_pending_escape_if_ready(after_redraw)

    assert keyboard.pressed == [Direction.RIGHT]
    assert any('"trigger": "food_redraw"' in event for event in emitted)


def test_queued_escape_is_prebuffered_when_head_reaches_trigger_phase() -> None:
    emitted: list[str] = []
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
        emit=emitted.append,
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(1, 7),
        target=Cell(0, 7),
        direction=Direction.DOWN,
        food_before=Cell(0, 8),
        eats=True,
        trigger_cell=Cell(0, 7),
    )
    entering = FrameObservation(
        geometry=BoardGeometry(0, 0, 595, 525, 35, 17, 15),
        snake_cells=frozenset({Cell(1, 7), Cell(0, 7)}),
        food=Cell(0, 8),
        confidence=1.0,
        head_hint=Cell(1, 7),
        head_center_px=(28.0, 262.5),
        head_confidence=1.0,
    )

    controller._send_pending_escape_if_ready(entering)

    assert keyboard.pressed == [Direction.DOWN]
    assert controller._pending_escape.sent
    assert any('"trigger": "head_phase"' in event for event in emitted)
    assert any('"head_phase": 0.7' in event for event in emitted)


def test_phase_buffer_arms_following_turns_before_next_cell() -> None:
    tracker = StateTracker()
    geometry = BoardGeometry(0, 0, 595, 525, 35, 17, 15)
    tracker._state = SnakeState(
        geometry, (Cell(7, 7), Cell(7, 6), Cell(7, 5)),
        Cell(6, 7), Direction.DOWN, 1,
    )
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=tracker,
        policy=None,  # type: ignore[arg-type]
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(7, 7),
        target=Cell(7, 8),
        direction=Direction.LEFT,
        food_before=Cell(6, 7),
        eats=False,
        travel_direction=Direction.DOWN,
        following_turn=Direction.UP,
        third_turn=Direction.RIGHT,
    )
    controller._send_pending_escape_if_ready(FrameObservation(
        geometry=geometry,
        snake_cells=frozenset({Cell(7, 8)}),
        food=Cell(6, 7),
        confidence=1.0,
        head_center_px=(262.5, 283.5),
    ))
    assert keyboard.pressed == [Direction.LEFT]
    assert controller._pending_escape.origin == Cell(7, 8)
    assert controller._pending_escape.target == Cell(6, 8)
    assert controller._pending_escape.direction == Direction.UP

    tracker._state = SnakeState(
        geometry, (Cell(7, 8), Cell(7, 7), Cell(7, 6)),
        Cell(6, 7), Direction.DOWN, 2,
    )
    controller._send_pending_escape_if_ready(FrameObservation(
        geometry=geometry,
        snake_cells=frozenset({Cell(6, 8)}),
        food=Cell(6, 7),
        confidence=1.0,
        head_center_px=(245.0, 297.5),
    ))
    assert keyboard.pressed == [Direction.LEFT, Direction.UP]
    assert controller._pending_escape.origin == Cell(6, 8)
    assert controller._pending_escape.target == Cell(6, 7)
    assert controller._pending_escape.direction == Direction.RIGHT


def test_straight_edge_approach_waits_for_safe_head_phase() -> None:
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(5, 12),
        target=Cell(5, 13),
        direction=Direction.RIGHT,
        food_before=Cell(5, 14),
        eats=True,
        trigger_cell=Cell(5, 14),
        trigger_on_target=False,
    )
    geometry = BoardGeometry(0, 0, 595, 525, 35, 17, 15)

    controller._send_pending_escape_if_ready(
        FrameObservation(
            geometry=geometry,
            snake_cells=frozenset({Cell(5, 12), Cell(5, 13), Cell(5, 14)}),
            food=Cell(5, 14),
            confidence=1.0,
            head_hint=Cell(5, 13),
            head_center_px=(192.5, 455.0),
            head_confidence=1.0,
        )
    )
    assert keyboard.pressed == []

    controller._send_pending_escape_if_ready(
        FrameObservation(
            geometry=geometry,
            snake_cells=frozenset({Cell(5, 13), Cell(5, 14)}),
            food=Cell(5, 14),
            confidence=1.0,
            head_hint=Cell(5, 13),
            head_center_px=(192.5, 472.5),
            head_confidence=1.0,
        )
    )
    assert keyboard.pressed == [Direction.RIGHT]


def test_stale_food_redraw_escape_is_cancelled_after_path_divergence() -> None:
    emitted: list[str] = []
    tracker = StateTracker()
    tracker._state = SnakeState(
        geometry=BoardGeometry(0, 0, 595, 525, 35, 17, 15),
        snake=(Cell(1, 14), Cell(1, 13), Cell(1, 12)),
        food=Cell(0, 14),
        direction=Direction.DOWN,
        frame_index=1119,
    )
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=tracker,
        policy=None,  # type: ignore[arg-type]
        emit=emitted.append,
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(1, 13),
        target=Cell(0, 13),
        direction=Direction.DOWN,
        food_before=Cell(0, 14),
        eats=True,
        trigger_cell=Cell(0, 13),
        travel_direction=Direction.LEFT,
    )
    redrawn = FrameObservation(
        geometry=tracker.state.geometry,
        snake_cells=frozenset({Cell(1, 14), Cell(0, 14)}),
        food=Cell(8, 6),
        confidence=1.0,
        head_hint=Cell(1, 14),
        head_center_px=(52.5, 507.5),
        head_confidence=1.0,
    )

    controller._send_pending_escape_if_ready(redrawn)

    assert keyboard.pressed == []
    assert controller._pending_escape is None
    assert any('"status": "queued_escape_cancelled"' in event for event in emitted)
    assert any('"reason": "path_diverged"' in event for event in emitted)


def test_straight_decision_does_not_send_redundant_key() -> None:
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    state = SnakeState(
        geometry=BoardGeometry(0, 0, 170, 150, 10, 17, 15),
        snake=(Cell(4, 7), Cell(3, 7), Cell(2, 7)),
        food=Cell(12, 7),
        direction=Direction.RIGHT,
        frame_index=1,
    )
    straight = Decision(
        probabilities={"UP": 0.0, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 1.0},
        proposed=Direction.RIGHT,
        executed=Direction.RIGHT,
        safe_directions=(Direction.UP, Direction.DOWN, Direction.RIGHT),
        intervened=False,
        inference_ms=1.0,
        planner_best=Direction.RIGHT,
    )

    controller._apply_decision(state, straight)

    assert keyboard.pressed == []
    assert controller._pending_turn is None


def test_unacknowledged_turn_is_retried_without_changing_direction() -> None:
    emitted: list[str] = []
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
        emit=emitted.append,
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_turn = _PendingTurn(
        origin=Cell(4, 7),
        from_direction=Direction.RIGHT,
        direction=Direction.DOWN,
    )
    continued = SnakeState(
        geometry=BoardGeometry(0, 0, 170, 150, 10, 17, 15),
        snake=(Cell(5, 7), Cell(4, 7), Cell(3, 7)),
        food=Cell(12, 7),
        direction=Direction.RIGHT,
        frame_index=2,
    )

    assert controller._handle_pending_turn(continued)
    assert keyboard.pressed == [Direction.DOWN]
    assert controller._pending_turn is not None
    assert controller._pending_turn.direction == Direction.DOWN
    assert controller._pending_turn.attempts == 2

    acknowledged = SnakeState(
        geometry=continued.geometry,
        snake=(Cell(5, 8), Cell(5, 7), Cell(4, 7)),
        food=Cell(12, 7),
        direction=Direction.DOWN,
        frame_index=3,
    )
    assert not controller._handle_pending_turn(acknowledged)
    assert controller._pending_turn is None
    assert any('"status": "turn_retried"' in event for event in emitted)
    assert any('"status": "turn_acknowledged"' in event for event in emitted)


def test_escape_after_phase_window_is_cancelled_not_sent() -> None:
    emitted: list[str] = []
    controller = LiveController(
        capture=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        tracker=StateTracker(),
        policy=None,  # type: ignore[arg-type]
        emit=emitted.append,
    )
    keyboard = FakeKeyboard()
    controller.keyboard = keyboard  # type: ignore[assignment]
    controller._pending_escape = _PendingEscape(
        origin=Cell(1, 7),
        target=Cell(0, 7),
        direction=Direction.DOWN,
        food_before=Cell(0, 8),
        eats=True,
        travel_direction=Direction.LEFT,
    )
    too_late = FrameObservation(
        geometry=BoardGeometry(0, 0, 595, 525, 35, 17, 15),
        snake_cells=frozenset({Cell(1, 7), Cell(0, 7)}),
        food=Cell(0, 8),
        confidence=1.0,
        head_hint=Cell(0, 7),
        head_center_px=(10.5, 262.5),
        head_confidence=1.0,
    )

    controller._send_pending_escape_if_ready(too_late)

    assert keyboard.pressed == []
    assert controller._pending_escape is None
    assert any('"reason": "phase_window_missed"' in event for event in emitted)
