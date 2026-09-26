from snake_laya.model import BoardGeometry, Cell, Direction, SnakeState
from snake_laya.planner import SafetyPlanner
from snake_laya.policy import LayaPolicy, build_move_request


GEOMETRY = BoardGeometry(0, 0, 170, 150, 10, 17, 15)


def state() -> SnakeState:
    return SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(4, 7), Cell(3, 7), Cell(2, 7)),
        food=Cell(12, 7),
        direction=Direction.RIGHT,
        frame_index=1,
    )


def test_planner_rejects_reverse_and_prefers_food_direction() -> None:
    planner = SafetyPlanner()
    moves = {move.direction: move for move in planner.analyse(state())}

    assert moves[Direction.LEFT].reason == "reverse"
    assert planner.preferred(tuple(moves.values())) == Direction.RIGHT


def test_planner_rejects_tight_turn_into_current_tail_cell() -> None:
    tight_turn = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(15, 8), Cell(16, 8), Cell(16, 7), Cell(15, 7)),
        food=Cell(16, 3),
        direction=Direction.LEFT,
        frame_index=126,
    )

    moves = {move.direction: move for move in SafetyPlanner().analyse(tight_turn)}

    assert not moves[Direction.UP].legal
    assert moves[Direction.UP].reason == "body"


def test_planner_rejects_apple_bite_with_no_legal_exit() -> None:
    trap = SnakeState(
        geometry=GEOMETRY,
        snake=tuple(Cell(*point) for point in (
            (12, 5), (12, 6), (13, 6), (14, 6), (15, 6),
            (15, 5), (15, 4), (14, 4), (13, 4),
        )),
        food=Cell(13, 5),
        direction=Direction.UP,
        frame_index=58,
    )

    bite = next(
        move for move in SafetyPlanner().analyse(trap)
        if move.direction == Direction.RIGHT
    )

    assert bite.legal
    assert not bite.safe
    assert bite.reason == "no legal exit"
    assert Direction.RIGHT not in build_move_request(trap).allowed


class FakeAgent:
    def predict(self, prompt, questions):
        assert "LEFT" not in questions["move"]["criteria"]
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.1,
                        "DOWN": 0.2,
                        "LEFT": 0.6,
                        "RIGHT": 0.1,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.8},
            }
        }


def test_policy_never_offers_reverse_choice_to_laya() -> None:
    decision = LayaPolicy(agent=FakeAgent()).decide(state())

    assert decision.proposed == Direction.DOWN
    assert decision.executed == Direction.DOWN
    assert not decision.intervened
    assert decision.probabilities["LEFT"] == 0.0


class SafeDetourAgent:
    def predict(self, prompt, questions):
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.8,
                        "DOWN": 0.1,
                        "LEFT": 0.05,
                        "RIGHT": 0.05,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.8},
            }
        }


def test_laya_owns_execution_when_its_proposal_is_safe() -> None:
    decision = LayaPolicy(agent=SafeDetourAgent()).decide(state())

    assert decision.proposed == Direction.UP
    assert Direction.UP in decision.safe_directions
    assert decision.planner_best == Direction.RIGHT
    assert decision.executed == Direction.UP
    assert not decision.intervened


class NearTieAgent:
    def predict(self, prompt, questions):
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.301,
                        "DOWN": 0.1,
                        "LEFT": 0.29,
                        "RIGHT": 0.3,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.8},
            }
        }


def test_near_tied_safe_detour_remains_layas_choice() -> None:
    decision = LayaPolicy(agent=NearTieAgent()).decide(state())

    assert decision.proposed == Direction.UP
    assert decision.planner_best == Direction.RIGHT
    assert decision.executed == Direction.UP
    assert not decision.intervened
    assert decision.intervention_reason is None


class CountingAgent:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, prompt, questions):
        self.calls += 1
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.1,
                        "DOWN": 0.1,
                        "LEFT": 0.0,
                        "RIGHT": 0.8,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.9},
            }
        }


def test_identical_laya_input_reuses_exact_model_output() -> None:
    agent = CountingAgent()
    policy = LayaPolicy(agent=agent)

    first = policy.decide(state())
    second = policy.decide(state())

    assert agent.calls == 1
    assert not first.cache_hit
    assert second.cache_hit
    assert second.probabilities == first.probabilities
    assert second.executed == first.executed


class FoodLookaheadAgent:
    def predict(self, prompt, questions):
        assert "eat food now" in questions["move"]["criteria"]["DOWN"]
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.0,
                        "DOWN": 0.99,
                        "LEFT": 0.0,
                        "RIGHT": 0.01,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.99},
            }
        }


def test_laya_chooses_imminent_food_turn_one_cell_early() -> None:
    approaching = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(7, 11), Cell(6, 11), Cell(5, 11), Cell(4, 11)),
        food=Cell(8, 12),
        direction=Direction.RIGHT,
        frame_index=280,
    )

    decision = LayaPolicy(agent=FoodLookaheadAgent()).decide(approaching)

    assert decision.proposed == Direction.RIGHT
    assert decision.executed == Direction.RIGHT
    assert decision.queued_escape == Direction.DOWN
    assert decision.intervention_reason == "laya_food_lookahead"
    assert decision.lookahead_head == Cell(8, 11)
    assert decision.lookahead_probabilities is not None
    assert decision.lookahead_probabilities["DOWN"] > 0.98


def test_planner_masks_head_on_corner_apple() -> None:
    planner = SafetyPlanner()
    near_top = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(12, 1), Cell(12, 2), Cell(12, 3), Cell(12, 4)),
        food=Cell(16, 0),
        direction=Direction.UP,
        frame_index=1,
    )
    assert planner.preferred(planner.analyse(near_top)) == Direction.RIGHT

    below_food = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(16, 1), Cell(15, 1), Cell(14, 1), Cell(13, 1)),
        food=Cell(16, 0),
        direction=Direction.RIGHT,
        frame_index=2,
    )
    moves = {move.direction: move for move in planner.analyse(below_food)}
    assert moves[Direction.UP].safe
    assert not moves[Direction.UP].operationally_safe
    assert moves[Direction.UP].corner
    assert planner.preferred(tuple(moves.values())) == Direction.DOWN


def test_non_eating_outer_wall_plan_has_escape_options() -> None:
    near_right_wall = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(15, 9), Cell(14, 9), Cell(13, 9)),
        food=Cell(4, 4),
        direction=Direction.RIGHT,
        frame_index=1,
    )

    request = build_move_request(near_right_wall)

    assert Direction.RIGHT in request.allowed
    assert set(request.escape_options[Direction.RIGHT]) == {
        Direction.UP,
        Direction.DOWN,
    }


class HeadOnEdgeAgent:
    def predict(self, prompt, questions):
        assert "Moving RIGHT" in prompt
        assert "LEFT" not in questions["move"]["criteria"]
        if "RIGHT" in questions["move"]["criteria"]:
            assert "queued escape UP" in questions["move"]["criteria"]["RIGHT"]
        answers = {
            "move": {
                "probabilities": {
                    "UP": 0.04,
                    "DOWN": 0.06,
                    "RIGHT": 0.9,
                }
            },
            "risk": {"noul": 0.9},
            "food": {"noul": 0.8},
        }
        for question_id, question in questions.items():
            if question_id.startswith("escape_"):
                answers[question_id] = {
                    "probabilities": {
                        key: (0.8 if key == "UP" else 0.2)
                        for key in question["criteria"]
                    }
                }
        return {
            "answers": answers,
        }


def test_head_on_edge_apple_gets_laya_selected_queued_escape() -> None:
    edge_state = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(15, 12), Cell(14, 12), Cell(13, 12), Cell(12, 12)),
        food=Cell(16, 12),
        direction=Direction.RIGHT,
        frame_index=1,
    )

    decision = LayaPolicy(agent=HeadOnEdgeAgent()).decide(edge_state)

    assert decision.proposed == Direction.RIGHT
    assert decision.executed == Direction.RIGHT
    assert decision.queued_escape == Direction.UP
    assert not decision.intervened


def test_non_eating_wall_plan_preserves_layas_queued_escape() -> None:
    boundary_transit = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(15, 12), Cell(14, 12), Cell(13, 12), Cell(12, 12)),
        food=Cell(4, 3),
        direction=Direction.RIGHT,
        frame_index=1,
    )

    decision = LayaPolicy(agent=HeadOnEdgeAgent()).decide(boundary_transit)

    assert decision.proposed == Direction.RIGHT
    assert decision.executed == Direction.RIGHT
    assert decision.queued_escape == Direction.UP
    assert not decision.intervened


class MustNotRunAgent:
    def predict(self, prompt, questions):
        raise AssertionError("forced-turn safety must not wait for Laya")


def test_forced_body_turn_bypasses_inference() -> None:
    forced_turn = SnakeState(
        geometry=GEOMETRY,
        snake=(
            Cell(12, 3), Cell(12, 2), Cell(11, 2), Cell(10, 2),
            Cell(10, 3), Cell(10, 4), Cell(11, 4), Cell(12, 4),
        ),
        food=Cell(16, 10),
        direction=Direction.DOWN,
        frame_index=492,
    )

    decision = LayaPolicy(agent=MustNotRunAgent()).decide(forced_turn)

    assert decision.executed == Direction.RIGHT
    assert decision.intervention_reason == "forced_turn_guard"
    assert decision.inference_ms == 0.0


class EdgeApproachAgent:
    def predict(self, prompt, questions):
        answers = {
            "move": {
                "probabilities": {
                    "DOWN": 0.99,
                    "LEFT": 0.005,
                    "RIGHT": 0.005,
                }
            },
            "risk": {"noul": 0.9},
            "food": {"noul": 0.9},
            "escape_DOWN": {
                "probabilities": {"LEFT": 0.1, "RIGHT": 0.9}
            },
        }
        return {"answers": answers}


def test_edge_apple_approach_keeps_layas_queued_escape() -> None:
    approach = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(5, 12), Cell(5, 11), Cell(5, 10)),
        food=Cell(5, 14),
        direction=Direction.DOWN,
        frame_index=1,
    )

    decision = LayaPolicy(agent=EdgeApproachAgent()).decide(approach)

    assert decision.proposed == Direction.DOWN
    assert decision.executed == Direction.DOWN
    assert decision.queued_escape == Direction.RIGHT
    assert not decision.intervened


class BodyMarginAgent:
    def predict(self, prompt, questions):
        assert "LEFT" not in questions["move"]["criteria"]
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.1,
                        "DOWN": 0.8,
                        "LEFT": 0.1,
                        "RIGHT": 0.0,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.8},
            }
        }


def test_laya_cannot_turn_into_a_one_cell_body_deadline() -> None:
    # Live run 215550: LEFT enters (14, 10), but the following LEFT cell is
    # the tail at (13, 10). The next screenshot is too late to turn away.
    near_body = SnakeState(
        geometry=GEOMETRY,
        snake=(
            Cell(15, 10), Cell(15, 9), Cell(15, 8), Cell(16, 8),
            Cell(16, 7), Cell(15, 7), Cell(14, 7), Cell(14, 8),
            Cell(13, 8), Cell(13, 9), Cell(13, 10), Cell(12, 10),
        ),
        food=Cell(9, 10),
        direction=Direction.DOWN,
        frame_index=658,
    )

    decision = LayaPolicy(agent=BodyMarginAgent()).decide(near_body)

    assert Direction.LEFT not in decision.safe_directions
    assert decision.executed == Direction.DOWN
    assert not decision.intervened


def test_body_two_cells_ahead_forces_an_early_turn() -> None:
    # Live run 215707: (4, 2) is free, but continuing to (3, 2) hits body.
    # The guard must turn at (5, 2), before another model round trip.
    body_two_ahead = SnakeState(
        geometry=GEOMETRY,
        snake=(
            Cell(5, 2), Cell(6, 2), Cell(6, 3), Cell(5, 3), Cell(4, 3),
            Cell(3, 3), Cell(3, 2), Cell(2, 2), Cell(2, 3),
        ),
        food=Cell(0, 1),
        direction=Direction.LEFT,
        frame_index=619,
    )

    decision = LayaPolicy(agent=MustNotRunAgent()).decide(body_two_ahead)

    assert Direction.LEFT not in decision.safe_directions
    assert decision.executed == Direction.UP
    assert decision.intervention_reason == "forced_turn_guard"
    assert decision.inference_ms == 0.0


def test_boxed_state_keeps_safe_one_cell_choices_for_laya() -> None:
    # Simulator seed 10003 previously raised despite two collision-free moves.
    boxed = SnakeState(
        geometry=GEOMETRY,
        snake=tuple(Cell(*point) for point in (
            (6, 5), (7, 5), (7, 4), (6, 4), (6, 3), (5, 3),
            (5, 2), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
            (4, 7), (5, 7), (6, 7), (7, 7),
        )),
        food=Cell(7, 6),
        direction=Direction.LEFT,
        frame_index=144,
    )

    request = build_move_request(boxed)

    assert set(request.allowed) == {Direction.DOWN, Direction.LEFT}


def test_sealed_corridor_offers_legal_continuation() -> None:
    sealed = SnakeState(
        geometry=GEOMETRY,
        snake=tuple(Cell(*point) for point in (
            (0, 8), (0, 7), (1, 7), (2, 7), (3, 7), (4, 7),
            (5, 7), (6, 7), (7, 7), (8, 7), (8, 6), (9, 6),
            (10, 6), (11, 6), (12, 6), (13, 6), (14, 6),
            (15, 6), (16, 6), (16, 5), (16, 4), (15, 4),
            (15, 5), (14, 5), (14, 4), (13, 4), (13, 3),
            (12, 3), (11, 3),
        )),
        food=Cell(5, 8),
        direction=Direction.DOWN,
        frame_index=349,
    )

    assert set(build_move_request(sealed).allowed) == {
        Direction.DOWN, Direction.RIGHT,
    }


def test_single_executable_move_skips_unsupported_one_choice_inference() -> None:
    corridor = SnakeState(
        geometry=GEOMETRY,
        snake=tuple(Cell(*point) for point in (
            (12, 2), (12, 3), (12, 4), (13, 4), (13, 3),
            (13, 2), (13, 1), (14, 1), (15, 1), (15, 2),
            (15, 3), (15, 4), (15, 5), (14, 5), (14, 6),
            (13, 6), (12, 6), (11, 6), (11, 5), (11, 4),
            (11, 3), (11, 2), (11, 1), (10, 1),
        )),
        food=Cell(10, 9),
        direction=Direction.UP,
        frame_index=277,
    )

    decision = LayaPolicy(agent=MustNotRunAgent()).decide(corridor)

    assert decision.executed == Direction.UP
    assert decision.intervention_reason == "single_executable_move"


def test_live_policy_turns_before_bottom_wall_with_one_cell_input_lag() -> None:
    approaching = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(6, 12), Cell(6, 11), Cell(6, 10)),
        food=Cell(10, 10),
        direction=Direction.DOWN,
        frame_index=1,
    )

    class DownAgent:
        def predict(self, prompt, questions):
            answers = {
                "move": {"probabilities": {
                    "UP": 0.0, "DOWN": 0.9, "LEFT": 0.05, "RIGHT": 0.05,
                }},
                "risk": {"noul": 0.9},
                "food": {"noul": 0.9},
            }
            for key, question in questions.items():
                if key.startswith("escape_"):
                    answers[key] = {"probabilities": {
                        direction: 1.0 / len(question["criteria"])
                        for direction in question["criteria"]
                    }}
            return {"answers": answers}

    decision = LayaPolicy(agent=DownAgent(), live_timing=True).decide(approaching)

    assert decision.proposed == Direction.DOWN
    assert decision.executed == Direction.RIGHT
    assert decision.intervention_reason == "preemptive_wall_guard"
    assert decision.queued_escape is None


def test_live_policy_excludes_turn_into_body_after_straight_step() -> None:
    delayed_trap = SnakeState(
        geometry=GEOMETRY,
        snake=tuple(Cell(*point) for point in (
            (5, 5), (4, 5), (3, 5), (3, 4), (3, 3),
            (3, 2), (4, 2), (4, 1), (5, 1), (6, 1),
            (7, 1), (7, 2), (7, 3), (7, 4), (6, 4),
            (6, 3), (6, 2),
        )),
        food=Cell(5, 0),
        direction=Direction.RIGHT,
        frame_index=1,
    )

    immediate = build_move_request(delayed_trap)
    delayed = build_move_request(
        delayed_trap, reaction_steps=2, delayed_turns=True
    )

    assert Direction.UP in immediate.allowed
    assert Direction.UP not in delayed.allowed


def test_live_laya_plan_buffers_two_adjacent_turns() -> None:
    approaching = SnakeState(
        geometry=GEOMETRY,
        snake=(Cell(7, 7), Cell(7, 6), Cell(7, 5)),
        food=Cell(6, 7),
        direction=Direction.DOWN,
        frame_index=1,
    )

    class LeftAgent:
        def predict(self, prompt, questions):
            answers = {
                "move": {"probabilities": {
                    "UP": 0.02, "DOWN": 0.02, "LEFT": 0.94, "RIGHT": 0.02,
                }},
                "risk": {"noul": 0.9},
                "food": {"noul": 0.9},
            }
            for key, question in questions.items():
                if key.startswith("escape_"):
                    answers[key] = {"probabilities": {
                        direction: 1.0 / len(question["criteria"])
                        for direction in question["criteria"]
                    }}
            return {"answers": answers}

    decision = LayaPolicy(agent=LeftAgent(), live_timing=True).decide_live(approaching)

    assert decision.executed == Direction.DOWN
    assert decision.queued_escape == Direction.LEFT
    assert decision.following_turn == Direction.UP
    assert decision.lookahead_head == Cell(7, 8)


def test_food_move_is_masked_when_growth_closes_the_reaction_cell() -> None:
    # Live run 215640: UP would eat at (3, 9), but growth retains the tail and
    # the following UP cell (3, 8) is still body. A late input cannot escape.
    growth_trap = SnakeState(
        geometry=GEOMETRY,
        snake=(
            Cell(3, 10), Cell(4, 10), Cell(4, 9), Cell(4, 8),
            Cell(3, 8), Cell(2, 8), Cell(2, 9),
        ),
        food=Cell(3, 9),
        direction=Direction.LEFT,
        frame_index=227,
    )

    request = build_move_request(growth_trap)

    assert Direction.UP not in request.allowed
    assert Direction.LEFT in request.allowed


class CornerSetupAgent:
    def predict(self, prompt, questions):
        if "RIGHT" in questions["move"]["criteria"]:
            assert "escape_RIGHT" in questions
            assert "eats food" in questions["escape_RIGHT"]["criteria"]["DOWN"]
            assert "does not eat food" in questions["escape_RIGHT"]["criteria"]["UP"]
        return {
            "answers": {
                "move": {
                    "probabilities": {
                        "UP": 0.0,
                        "DOWN": 0.3,
                        "LEFT": 0.0,
                        "RIGHT": 0.7,
                    }
                },
                "risk": {"noul": 0.9},
                "food": {"noul": 0.9},
                "escape_RIGHT": {
                    "probabilities": {"UP": 0.1, "DOWN": 0.9}
                },
            }
        }


def test_perpendicular_edge_setup_for_corner_food_is_not_overridden() -> None:
    # Live run 220918: enter (16, 13), then use the queued DOWN turn to eat
    # the corner apple at (16, 14). This is purposeful boundary travel.
    corner_setup = SnakeState(
        geometry=GEOMETRY,
        snake=(
            Cell(15, 13), Cell(15, 12), Cell(15, 11), Cell(15, 10),
            Cell(15, 9),
        ),
        food=Cell(16, 14),
        direction=Direction.DOWN,
        frame_index=552,
    )

    decision = LayaPolicy(agent=CornerSetupAgent()).decide(corner_setup)

    assert decision.proposed == Direction.RIGHT
    assert decision.executed == Direction.RIGHT
    assert decision.queued_escape == Direction.DOWN
    assert not decision.intervened


class CornerContinuationAgent:
    def predict(self, prompt, questions):
        if "LEFT" in questions["move"]["criteria"]:
            assert "escape_LEFT" in questions
            assert "continue UP to eat food" in questions["escape_LEFT"]["criteria"]["UP"]
        answers = {
            "move": {
                "probabilities": {
                    "UP": 0.01,
                    "DOWN": 0.01,
                    "LEFT": 0.97,
                    "RIGHT": 0.01,
                }
            },
            "risk": {"noul": 0.9},
            "food": {"noul": 0.9},
        }
        for question_id, question in questions.items():
            if question_id.startswith("escape_"):
                answers[question_id] = {
                    "probabilities": {
                        key: (0.99 if key == "UP" else 0.01)
                        for key in question["criteria"]
                    }
                }
        return {"answers": answers}


def test_corner_food_two_cells_after_queued_turn_is_not_overridden() -> None:
    # Live run 165946: from (1, 2), LEFT then queued UP reaches the outer
    # column safely, and continuing UP eats the corner apple at (0, 0).
    corner_approach = SnakeState(
        geometry=GEOMETRY,
        snake=(
            Cell(1, 2), Cell(1, 3), Cell(2, 3),
            Cell(3, 3), Cell(4, 3), Cell(4, 2),
        ),
        food=Cell(0, 0),
        direction=Direction.UP,
        frame_index=527,
    )

    decision = LayaPolicy(agent=CornerContinuationAgent()).decide(corner_approach)

    assert decision.proposed == Direction.LEFT
    assert decision.executed == Direction.LEFT
    assert decision.queued_escape == Direction.UP
    assert not decision.intervened
    assert decision.intervention_reason is None
