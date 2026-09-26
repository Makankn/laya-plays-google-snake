from __future__ import annotations

from snake_laya.overlay import JsonlTail, OverlayTelemetry, place_away_from_board
from snake_laya.cli import build_parser


def test_live_overlay_is_opt_in():
    assert build_parser().parse_args(["live"]).overlay is False
    assert build_parser().parse_args(["live", "--overlay"]).overlay is True


def test_overlay_starts_outside_saved_capture_region():
    board = (1582, 399, 670, 671)
    x, y = place_away_from_board(board, (2560, 1440))
    assert x == 1212
    assert y == 399
    assert x + 354 <= board[0]


def test_tail_waits_for_complete_lines(tmp_path):
    path = tmp_path / "run.jsonl"
    tail = JsonlTail(path)
    assert tail.read() == []
    path.write_bytes(b'{"status":"watching"}\n{"status":"dec')
    assert tail.read() == [{"status": "watching"}]
    with path.open("ab") as stream:
        stream.write(b'ision"}\n')
    assert tail.read() == [{"status": "decision"}]
    assert tail.read() == []


def test_decision_shows_laya_probabilities_and_visual_confidence():
    telemetry = OverlayTelemetry()
    telemetry.ingest({
        "status": "decision",
        "state": {"direction": "RIGHT"},
        "decision": {
            "probabilities": {"UP": 0.1, "RIGHT": 0.6, "DOWN": 0.2, "LEFT": 0.1},
            "executed": "RIGHT", "queued_escape": "DOWN", "inference_ms": 7.4,
            "intervened": False,
        },
        "perception": {"board_confidence": 0.96, "head_confidence": 0.81},
    })
    assert telemetry.direction == "DOWN"
    assert telemetry.current_direction == "RIGHT"
    assert telemetry.probabilities == {"UP": 0.1, "RIGHT": 0.6, "DOWN": 0.2, "LEFT": 0.1}
    assert telemetry.latency_ms == 7.4
    assert telemetry.board_confidence == 0.96
    assert telemetry.head_confidence == 0.81


def test_safety_one_hot_is_not_misreported_as_laya_confidence():
    telemetry = OverlayTelemetry()
    telemetry.ingest({
        "status": "decision", "state": {},
        "decision": {
            "probabilities": {"UP": 0, "RIGHT": 1, "DOWN": 0, "LEFT": 0},
            "executed": "RIGHT", "intervened": True,
            "intervention_reason": "forced_turn_guard",
        },
    })
    assert telemetry.probabilities is None
    assert telemetry.probability_label == "SAFETY-ONLY TURN"


def test_food_lookahead_uses_actual_model_distribution():
    telemetry = OverlayTelemetry()
    telemetry.ingest({
        "status": "decision", "state": {},
        "decision": {
            "probabilities": {"UP": 0, "RIGHT": 1, "DOWN": 0, "LEFT": 0},
            "lookahead_probabilities": {"UP": 0.2, "RIGHT": 0.3, "DOWN": 0.4, "LEFT": 0.1},
            "executed": "RIGHT", "intervention_reason": "laya_food_lookahead",
        },
    })
    assert telemetry.probabilities["DOWN"] == 0.4
    assert telemetry.probability_label == "LAYA · LOOKAHEAD"


def test_turn_acknowledgement_is_visible():
    telemetry = OverlayTelemetry()
    telemetry.ingest({"status": "turn_acknowledged", "direction": "LEFT", "attempts": 2})
    assert "after 2 tries" in telemetry.input_status


def test_lost_board_hides_stale_move_and_confidence():
    telemetry = OverlayTelemetry()
    telemetry.ingest({
        "status": "decision",
        "state": {"direction": "UP"},
        "decision": {
            "probabilities": {"UP": 0.16, "RIGHT": 0, "DOWN": 0, "LEFT": 0.84},
            "executed": "UP", "inference_ms": 50,
        },
        "perception": {"board_confidence": 0.45, "head_confidence": 1.0},
    })
    telemetry.ingest({"status": "waiting_for_board"})
    assert telemetry.status == "WAITING"
    assert telemetry.direction is None
    assert telemetry.probabilities is None
    assert telemetry.latency_ms is None
    assert telemetry.board_confidence is None
    assert telemetry.head_confidence is None
