from snake_laya.model import BoardGeometry, Cell, Direction, FrameObservation
from snake_laya.tracker import StateTracker


GEOMETRY = BoardGeometry(0, 0, 170, 150, 10, 17, 15)


def observation(
    *cells: tuple[int, int],
    food: tuple[int, int] | None = (12, 7),
    head: tuple[int, int] | None = None,
) -> FrameObservation:
    return FrameObservation(
        geometry=GEOMETRY,
        snake_cells=frozenset(Cell(x, y) for x, y in cells),
        food=None if food is None else Cell(*food),
        confidence=1.0,
        head_hint=None if head is None else Cell(*head),
    )


def test_tracker_bootstraps_head_and_direction_from_motion() -> None:
    tracker = StateTracker()
    assert tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7))) is None

    state = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))

    assert state is not None
    assert state.head == Cell(5, 7)
    assert state.direction == "RIGHT"
    assert state.snake == (Cell(5, 7), Cell(4, 7), Cell(3, 7))


def test_tracker_handles_growth_frame() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), food=(6, 7), head=(4, 7)))
    tracker.update(observation((3, 7), (4, 7), (5, 7), food=(6, 7), head=(5, 7)))

    state = tracker.update(observation((3, 7), (4, 7), (5, 7), (6, 7), food=None, head=(6, 7)))

    assert state is not None
    assert state.head == Cell(6, 7)
    assert len(state.snake) == 4


def test_animated_extra_cells_do_not_change_logical_length() -> None:
    tracker = StateTracker(initial_length=3)
    tracker.update(observation((1, 7), (2, 7), (3, 7), (4, 7), head=(4, 7)))

    first = tracker.update(observation((2, 7), (3, 7), (4, 7), (5, 7), head=(5, 7)))
    second = tracker.update(observation((3, 7), (4, 7), (5, 7), (6, 7), head=(6, 7)))

    assert first is not None and len(first.snake) == 3
    assert second is not None and len(second.snake) == 3
    assert second.snake == (Cell(6, 7), Cell(5, 7), Cell(4, 7))


def test_non_adjacent_head_fails_closed() -> None:
    tracker = StateTracker(initial_length=3)
    tracker.update(observation((1, 7), (2, 7), (3, 7), (4, 7), head=(4, 7)))
    assert tracker.update(observation((2, 7), (3, 7), (4, 7), (5, 7), head=(5, 7))) is not None

    bad = observation((2, 7), (3, 7), (4, 7), (8, 7), head=(8, 7))
    assert tracker.update(bad) is None
    assert tracker.desynchronized
    assert tracker.state is None
    assert tracker.desync_reason is not None
    assert "jump" in tracker.desync_reason


def test_blue_leading_cap_does_not_override_eye_position() -> None:
    tracker = StateTracker(initial_length=3)
    tracker.update(observation((2, 7), (3, 7), (4, 7), (5, 7), head=(4, 7)))
    tracker.update(observation((3, 7), (4, 7), (5, 7), (6, 7), head=(5, 7)))
    state = tracker.update(observation((4, 7), (5, 7), (6, 7), (7, 7), head=(6, 7)))

    assert state is not None
    assert state.head == Cell(6, 7)
    assert state.snake == (Cell(6, 7), Cell(5, 7), Cell(4, 7))
    assert not tracker.desynchronized


def test_missing_eye_frame_does_not_guess_from_blue_edge() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    state = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))
    assert state is not None
    advanced = tracker.update(observation((4, 7), (5, 7), (6, 7), head=None))
    assert advanced is not None
    assert advanced.head == Cell(5, 7)
    assert advanced.snake == (Cell(5, 7), Cell(4, 7), Cell(3, 7))


def test_new_food_is_recorded_without_a_second_head_move() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), food=(5, 7), head=(4, 7)))
    eaten = tracker.update(observation((3, 7), (4, 7), (5, 7), food=None, head=(5, 7)))
    assert eaten is not None and eaten.food is None

    refreshed = tracker.update(
        observation((3, 7), (4, 7), (5, 7), food=(10, 4), head=(5, 7))
    )
    assert refreshed is not None
    assert refreshed.head == Cell(5, 7)
    assert refreshed.food == Cell(10, 4)
    assert len(refreshed.snake) == 4
    assert refreshed.frame_index == eaten.frame_index


def test_straight_two_cell_eye_jump_interpolates_midpoint_and_growth() -> None:
    tracker = StateTracker()
    tracker.update(observation((1, 8), (2, 8), (3, 8), food=(4, 8), head=(3, 8)))
    tracker.update(
        observation((2, 8), (3, 8), (4, 8), food=(11, 2), head=(3, 8))
    )

    state = tracker.update(
        observation((3, 8), (4, 8), (5, 8), (6, 8), food=(11, 2), head=(5, 8))
    )

    assert state is not None
    assert state.head == Cell(5, 8)
    assert state.direction == Direction.RIGHT
    assert state.snake == (Cell(5, 8), Cell(4, 8), Cell(3, 8), Cell(2, 8))
    assert state.food == Cell(11, 2)
    assert tracker.transition_source == "confirmed_eyes_interpolated"
    assert not tracker.desynchronized


def test_three_cell_straight_jump_uses_visible_body_path() -> None:
    tracker = StateTracker(initial_length=7)
    tracker.update(observation((6, 9), head=(6, 9), food=(4, 5)))
    assert tracker.update(
        observation((5, 9), (6, 9), head=(5, 9), food=(4, 5))
    ) is not None

    state = tracker.update(
        observation(
            (2, 9), (3, 9), (4, 9), (5, 9),
            head=(2, 9),
            food=(4, 5),
        )
    )

    assert state is not None
    assert state.head == Cell(2, 9)
    assert state.direction == Direction.LEFT
    assert tracker.transition_source == "confirmed_eyes_interpolated"
    assert not tracker.desynchronized


def test_three_cell_straight_jump_without_blue_path_fails_closed() -> None:
    tracker = StateTracker()
    tracker.update(observation((4, 7), head=(4, 7)))
    tracker.update(observation((4, 7), (5, 7), head=(5, 7)))

    assert tracker.update(observation((8, 7), head=(8, 7))) is None
    assert tracker.desynchronized
    assert "unsupported straight eye jump" in (tracker.desync_reason or "")


def test_diagonal_two_cell_eye_jump_still_fails_closed() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    assert tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7))) is not None

    assert tracker.update(observation((4, 7), (5, 7), (6, 8), head=(6, 8))) is None
    assert tracker.desynchronized


def test_diagonal_jump_uses_unique_occupied_midpoint() -> None:
    tracker = StateTracker()
    tracker.update(observation((8, 7), (9, 7), (10, 7), head=(8, 7)))
    assert tracker.update(observation((7, 7), (8, 7), (9, 7), head=(7, 7))) is not None

    state = tracker.update(
        observation((6, 6), (6, 7), (7, 7), (8, 7), head=(6, 6))
    )

    assert state is not None
    assert state.head == Cell(6, 6)
    assert state.snake == (Cell(6, 6), Cell(6, 7), Cell(7, 7))
    assert state.direction == Direction.UP
    assert tracker.transition_source == "confirmed_eyes_interpolated_turn"
    assert not tracker.desynchronized


def test_three_cell_corner_jump_uses_unique_visible_path() -> None:
    tracker = StateTracker(initial_length=3)
    tracker.update(observation((12, 11), head=(12, 11), food=(4, 5)))
    assert tracker.update(
        observation((11, 11), (12, 11), head=(11, 11), food=(4, 5))
    ) is not None

    state = tracker.update(
        observation(
            (11, 12), (10, 12), (9, 12),
            head=(9, 12),
            food=(4, 5),
        )
    )

    assert state is not None
    assert state.head == Cell(9, 12)
    assert state.direction == Direction.LEFT
    assert state.snake == (
        Cell(9, 12), Cell(10, 12), Cell(11, 12)
    )
    assert tracker.transition_source == "confirmed_eyes_interpolated_corner"
    assert not tracker.desynchronized


def test_three_cell_corner_jump_without_unique_blue_path_fails_closed() -> None:
    tracker = StateTracker(initial_length=3)
    tracker.update(observation((5, 13), head=(5, 13), food=(4, 5)))
    assert tracker.update(
        observation((5, 12), (5, 13), head=(5, 12), food=(4, 5))
    ) is not None

    assert tracker.update(
        observation(
            (6, 12), (6, 11), (5, 11), (5, 10),
            head=(6, 10),
            food=(4, 5),
        )
    ) is None
    assert tracker.desynchronized
    assert "ambiguous corner eye jump" in (tracker.desync_reason or "")


def test_eye_animation_cannot_reverse_into_neck() -> None:
    tracker = StateTracker(confirmation_frames=2)
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))
    state = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))
    assert state is not None
    tracker.expect(Direction.UP)

    unchanged = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(4, 7)))
    assert unchanged == state
    assert len(set(unchanged.snake)) == len(unchanged.snake)
    assert not tracker.desynchronized


def test_confirmed_reverse_animation_is_rejected_then_tracking_recovers() -> None:
    tracker = StateTracker(confirmation_frames=1)
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    state = tracker.update(
        observation((3, 7), (4, 7), (5, 7), head=(5, 7))
    )
    assert state is not None

    unchanged = tracker.update(
        observation((3, 7), (4, 7), (5, 7), head=(4, 7))
    )

    assert unchanged == state
    assert tracker.rejected_hints == 1
    assert not tracker.desynchronized

    recovered = tracker.update(
        observation((4, 7), (5, 7), (6, 7), head=(6, 7))
    )
    assert recovered is not None
    assert recovered.head == Cell(6, 7)
    assert recovered.direction == Direction.RIGHT
    assert tracker.rejected_hints == 0
    assert not tracker.desynchronized


def test_visual_motion_is_authoritative_over_expected_command() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    state = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))
    assert state is not None
    tracker.expect(Direction.UP)

    continued = tracker.update(observation((4, 7), (5, 7), (6, 7), head=(6, 7)))
    assert continued is not None
    assert continued.head == Cell(6, 7)
    assert continued.direction == Direction.RIGHT


def test_apple_redraw_does_not_advance_head_before_eyes() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), food=(6, 7), head=(4, 7)))
    tracker.update(observation((3, 7), (4, 7), (5, 7), food=(6, 7), head=(5, 7)))
    tracker.expect(Direction.RIGHT)

    early = tracker.update(
        observation((4, 7), (5, 7), (6, 7), food=(10, 2), head=(5, 7))
    )
    assert early is not None
    assert early.head == Cell(5, 7)
    # The replacement is held separately until eyes confirm the old apple.
    assert early.food == Cell(6, 7)
    assert len(early.snake) == 3

    caught_up = tracker.update(
        observation((4, 7), (5, 7), (6, 7), food=(10, 2), head=(6, 7))
    )
    assert caught_up is not None
    assert caught_up.head == Cell(6, 7)
    assert caught_up.food == Cell(10, 2)
    assert len(caught_up.snake) == 4


def test_blue_leading_edge_never_advances_before_eyes() -> None:
    tracker = StateTracker()
    tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7)))
    state = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))
    assert state is not None
    tracker.expect(Direction.RIGHT)

    early = tracker.update(
        observation((4, 7), (5, 7), (6, 7), head=(5, 7))
    )
    assert early is not None
    assert early.head == Cell(5, 7)
    assert early.direction == Direction.RIGHT
    assert early.snake == (Cell(5, 7), Cell(4, 7), Cell(3, 7))


def test_adjacent_eye_position_requires_two_matching_frames() -> None:
    tracker = StateTracker(confirmation_frames=2)
    assert tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7))) is None
    assert tracker.update(observation((2, 7), (3, 7), (4, 7), head=(4, 7))) is None

    assert tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7))) is None
    state = tracker.update(observation((3, 7), (4, 7), (5, 7), head=(5, 7)))

    assert state is not None
    assert state.head == Cell(5, 7)
    assert state.direction == Direction.RIGHT
