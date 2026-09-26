from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from PIL import Image

from .capture import ScreenCapture
from .keyboard import WindowsKeyboard
from .model import Cell, Direction, FrameObservation, SnakeState
from .parser import ParseError, SnakeFrameParser
from .planner import VECTORS, SafetyPlanner
from .policy import Decision
from .tracker import StateTracker


class DecisionPolicy(Protocol):
    def decide(self, state: SnakeState) -> Decision: ...


@dataclass
class _PendingEscape:
    origin: Cell
    target: Cell
    direction: Direction
    food_before: Cell | None
    eats: bool
    trigger_cell: Cell | None = None
    trigger_on_target: bool = True
    travel_direction: Direction | None = None
    predecessor: Cell | None = None
    following_turn: Direction | None = None
    third_turn: Direction | None = None
    sent: bool = False


@dataclass
class _PendingTurn:
    origin: Cell
    from_direction: Direction
    direction: Direction
    attempts: int = 1


class LiveController:
    def __init__(
        self,
        capture: ScreenCapture,
        parser: SnakeFrameParser,
        tracker: StateTracker,
        policy: DecisionPolicy,
        *,
        execute: bool = False,
        fps: float = 30.0,
        duration: float | None = None,
        emit: Callable[[str], None] = print,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        self.capture = capture
        self.parser = parser
        self.tracker = tracker
        self.policy = policy
        self.keyboard = WindowsKeyboard() if execute else None
        self.period = 1.0 / fps
        self.duration = duration
        self.emit = emit
        self._last_decided_frame = -1
        self._last_wait_detail: str | None = None
        self._consecutive_parse_failures = 0
        self._desync_reported = False
        self._last_rejected_hints = 0
        self._board_seen = False
        self._pending_escape: _PendingEscape | None = None
        self._pending_turn: _PendingTurn | None = None
        self._planner = SafetyPlanner()

    def run(self) -> None:
        self.emit(json.dumps({"status": "watching", "keyboard": self.keyboard is not None}))
        active_started: float | None = None
        with self.capture:
            while self.duration is None or active_started is None or time.perf_counter() - active_started < self.duration:
                started = time.perf_counter()
                self.tick()
                if active_started is None and self._board_seen:
                    active_started = time.perf_counter()
                    self.emit(json.dumps({"status": "board_locked", "duration_started": self.duration}))
                remaining = self.period - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        self.emit(json.dumps({"status": "stopped", "reason": "duration_complete"}))

    def tick(self) -> Decision | None:
        frame = self.capture.grab()
        try:
            observation = self.parser.parse_array(frame)
        except ParseError as error:
            self._consecutive_parse_failures += 1
            if self._consecutive_parse_failures >= 3:
                self.tracker.mark_desynchronized(
                    f"{self._consecutive_parse_failures} consecutive unreadable frames: {error}"
                )
            detail = str(error)
            if detail != self._last_wait_detail:
                self.emit(json.dumps({"status": "waiting_for_board", "detail": detail}))
                self._last_wait_detail = detail
            return None
        self._consecutive_parse_failures = 0
        self._last_wait_detail = None
        self._board_seen = True
        if (
            observation.head_hint is None
            or observation.head_confidence < self.tracker.minimum_head_confidence
        ):
            self.emit(
                json.dumps(
                    {
                        "status": "eye_observation_rejected",
                        "reason": (
                            "missing"
                            if observation.head_hint is None
                            else "low_confidence"
                        ),
                        "head_hint": (
                            None
                            if observation.head_hint is None
                            else observation.head_hint.as_list()
                        ),
                        "head_confidence": round(observation.head_confidence, 4),
                        "tracked_head": (
                            None
                            if self.tracker.state is None
                            else self.tracker.state.head.as_list()
                        ),
                    }
                )
            )
        self._send_pending_escape_if_ready(observation)
        state = self.tracker.update(observation)
        if state is not None and hasattr(self.policy, "observe"):
            self.policy.observe(state)
        if self.tracker.rejected_hints != self._last_rejected_hints:
            if self.tracker.rejected_hints:
                self.emit(
                    json.dumps(
                        {
                            "status": "visual_hint_ignored",
                            "head_hint": None
                            if observation.head_hint is None
                            else observation.head_hint.as_list(),
                            "tracked_head": None
                            if self.tracker.state is None
                            else self.tracker.state.head.as_list(),
                            "consecutive": self.tracker.rejected_hints,
                        }
                    )
                )
            self._last_rejected_hints = self.tracker.rejected_hints
        if self.tracker.desynchronized and not self._desync_reported:
            debug_path = Path("captures") / f"desync-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(debug_path)
            self.emit(
                json.dumps(
                    {
                        "status": "desynchronized",
                        "detail": self.tracker.desync_reason,
                        "observation": observation.as_dict(),
                        "frame": str(debug_path.resolve()),
                        "action": "restart the controller before enabling keys",
                    }
                )
            )
            self._desync_reported = True
        if self._pending_escape is not None:
            pending = self._pending_escape
            if state is None:
                return None
            if pending.sent:
                if state.head not in self._pending_path(pending):
                    self._pending_escape = None
            elif state.head in self._pending_path(pending):
                if state.head != pending.origin:
                    self._last_decided_frame = state.frame_index
                return None
            else:
                self._cancel_pending_escape("path_diverged", state.head)
        if self._pending_turn is not None:
            if state is None:
                return None
            if self._handle_pending_turn(state):
                return None
            if self._pending_escape is not None and self._pending_escape.sent:
                self._pending_escape = None
        if state is None or state.food is None or state.frame_index == self._last_decided_frame:
            return None
        decision = (
            self.policy.decide_live(state)
            if hasattr(self.policy, "decide_live")
            else self.policy.decide(state)
        )
        self._apply_decision(state, decision, observation)
        return decision

    def _apply_decision(
        self,
        state: SnakeState,
        decision: Decision,
        observation: FrameObservation | None = None,
    ) -> None:
        self._last_decided_frame = state.frame_index
        key_sent = False
        if self.keyboard is not None:
            if self._pending_escape is not None:
                if self._pending_escape.sent:
                    self._pending_escape = None
                else:
                    self._cancel_pending_escape("superseded", state.head)
            if decision.executed != state.direction:
                self.keyboard.press(decision.executed)
                self.tracker.expect(decision.executed)
                self._pending_turn = _PendingTurn(
                    origin=state.head,
                    from_direction=state.direction,
                    direction=decision.executed,
                )
                key_sent = True
                self.emit(
                    json.dumps(
                        {
                            "status": "turn_requested",
                            "origin": state.head.as_list(),
                            "from_direction": state.direction.value,
                            "direction": decision.executed.value,
                            "attempt": 1,
                        }
                    )
                )
            if decision.queued_escape is not None:
                dx, dy = VECTORS[decision.executed]
                target = Cell(state.head.x + dx, state.head.y + dy)
                next_target = Cell(target.x + dx, target.y + dy)
                approaches_edge_food = state.food == next_target
                queued_dx, queued_dy = VECTORS[decision.queued_escape]
                queued_target = Cell(target.x + queued_dx, target.y + queued_dy)
                trigger_cell = next_target if approaches_edge_food else target
                self._pending_escape = _PendingEscape(
                    origin=state.head,
                    target=target,
                    direction=decision.queued_escape,
                    food_before=state.food,
                    eats=(
                        target == state.food
                        or approaches_edge_food
                        or queued_target == state.food
                    ),
                    trigger_cell=trigger_cell,
                    trigger_on_target=not approaches_edge_food,
                    travel_direction=decision.executed,
                    following_turn=decision.following_turn,
                    third_turn=decision.third_turn,
                )
        self.emit(
            json.dumps(
                {
                    "status": "decision",
                    "state": state.as_dict(),
                    "decision": decision.as_dict(),
                    "perception": None if observation is None else {
                        "board_confidence": round(observation.confidence, 4),
                        "head_confidence": round(observation.head_confidence, 4),
                    },
                    "transition_source": self.tracker.transition_source,
                    "keyboard_sent": key_sent,
                }
            )
        )
    def _send_pending_escape_if_ready(self, observation: FrameObservation) -> None:
        pending = self._pending_escape
        if pending is None or pending.sent or self.keyboard is None:
            return
        tracked_head = None if self.tracker.state is None else self.tracker.state.head
        if tracked_head is not None and tracked_head not in self._pending_path(pending):
            self._cancel_pending_escape("path_diverged", tracked_head)
            return
        trigger_cell = pending.trigger_cell or pending.target
        blue_entered = trigger_cell in observation.snake_cells
        head_phase = self._head_phase(observation, pending)
        trigger_steps = (
            abs(trigger_cell.x - pending.origin.x)
            + abs(trigger_cell.y - pending.origin.y)
        )
        phase_threshold = 0.15 if trigger_steps == 1 else trigger_steps - 1.25
        phase_maximum = 0.95 if trigger_steps == 1 else trigger_steps - 0.45
        phase_in_window = (
            head_phase is not None
            and phase_threshold <= head_phase <= phase_maximum
        )
        phase_ready = (
            blue_entered
            and phase_in_window
        )
        eye_entered = (
            pending.trigger_on_target
            and observation.head_hint == trigger_cell
            and observation.head_confidence >= 0.55
            and head_phase is None
        )
        food_redrawn = (
            pending.eats
            and observation.food != pending.food_before
            and phase_in_window
        )
        if head_phase is not None and head_phase > phase_maximum:
            self._cancel_pending_escape("phase_window_missed", tracked_head)
            return
        if not phase_ready and not eye_entered and not food_redrawn:
            return
        self.keyboard.press(pending.direction)
        self.tracker.expect(pending.direction)
        pending.sent = True
        current_direction = (
            None if self.tracker.state is None else self.tracker.state.direction
        )
        if current_direction != pending.direction:
            self._pending_turn = _PendingTurn(
                origin=(
                    pending.origin
                    if self.tracker.state is None
                    else self.tracker.state.head
                ),
                from_direction=(
                    pending.travel_direction
                    if current_direction is None
                    else current_direction
                ),
                direction=pending.direction,
            )
        self.emit(
            json.dumps(
                {
                    "status": "queued_escape_sent",
                    "target": pending.target.as_list(),
                    "trigger_cell": trigger_cell.as_list(),
                    "direction": pending.direction.value,
                    "trigger": (
                        "head_phase"
                        if phase_ready
                        else "food_redraw"
                        if food_redrawn
                        else "confirmed_eyes"
                    ),
                    "head_phase": None if head_phase is None else round(head_phase, 3),
                    "phase_threshold": round(phase_threshold, 3),
                    "phase_maximum": round(phase_maximum, 3),
                    "tracked_head": (
                        None if tracked_head is None else tracked_head.as_list()
                    ),
                }
            )
        )
        if pending.following_turn is not None:
            dx, dy = VECTORS[pending.direction]
            next_target = Cell(pending.target.x + dx, pending.target.y + dy)
            self._pending_escape = _PendingEscape(
                origin=pending.target,
                target=next_target,
                direction=pending.following_turn,
                food_before=observation.food,
                eats=(
                    observation.food == next_target
                    or observation.food == Cell(
                        next_target.x + VECTORS[pending.following_turn][0],
                        next_target.y + VECTORS[pending.following_turn][1],
                    )
                ),
                travel_direction=pending.direction,
                predecessor=pending.origin,
                following_turn=pending.third_turn,
            )
            self.emit(json.dumps({
                "status": "queued_followup_armed",
                "origin": pending.target.as_list(),
                "target": next_target.as_list(),
                "direction": pending.following_turn.value,
            }))

    def _handle_pending_turn(self, state: SnakeState) -> bool:
        pending = self._pending_turn
        if pending is None:
            return False
        if state.direction == pending.direction:
            self.emit(
                json.dumps(
                    {
                        "status": "turn_acknowledged",
                        "origin": pending.origin.as_list(),
                        "direction": pending.direction.value,
                        "head": state.head.as_list(),
                        "attempts": pending.attempts,
                    }
                )
            )
            self._pending_turn = None
            if self._pending_escape is not None and self._pending_escape.sent:
                self._pending_escape = None
            return False
        if state.head == pending.origin:
            return True
        move = next(
            item
            for item in self._planner.analyse(state)
            if item.direction == pending.direction
        )
        if not move.safe:
            self.emit(
                json.dumps(
                    {
                        "status": "turn_cancelled",
                        "reason": "requested_turn_became_unsafe",
                        "origin": pending.origin.as_list(),
                        "direction": pending.direction.value,
                        "head": state.head.as_list(),
                        "attempts": pending.attempts,
                    }
                )
            )
            self._pending_turn = None
            return False
        if self.keyboard is None:
            self._pending_turn = None
            return False
        self.keyboard.press(pending.direction)
        self.tracker.expect(pending.direction)
        pending.origin = state.head
        pending.from_direction = state.direction
        pending.attempts += 1
        self._last_decided_frame = state.frame_index
        self.emit(
            json.dumps(
                {
                    "status": "turn_retried",
                    "origin": state.head.as_list(),
                    "from_direction": state.direction.value,
                    "direction": pending.direction.value,
                    "attempt": pending.attempts,
                }
            )
        )
        return True

    @staticmethod
    def _pending_path(pending: _PendingEscape) -> tuple[Cell, ...]:
        trigger_cell = pending.trigger_cell or pending.target
        if trigger_cell == pending.target:
            path = (pending.origin, pending.target)
        else:
            path = (pending.origin, pending.target, trigger_cell)
        return path if pending.predecessor is None else (pending.predecessor, *path)

    @staticmethod
    def _head_phase(
        observation: FrameObservation, pending: _PendingEscape
    ) -> float | None:
        if observation.head_center_px is None:
            return None
        direction = pending.travel_direction
        if direction is None:
            direction = Direction.between(pending.origin, pending.target)
        dx, dy = VECTORS[direction]
        size = observation.geometry.cell_size_px
        origin_x = (pending.origin.x + 0.5) * size
        origin_y = (pending.origin.y + 0.5) * size
        center_x, center_y = observation.head_center_px
        return ((center_x - origin_x) * dx + (center_y - origin_y) * dy) / size

    def _cancel_pending_escape(self, reason: str, observed_head: Cell | None) -> None:
        pending = self._pending_escape
        if pending is None:
            return
        self.emit(
            json.dumps(
                {
                    "status": "queued_escape_cancelled",
                    "reason": reason,
                    "origin": pending.origin.as_list(),
                    "target": pending.target.as_list(),
                    "trigger_cell": (
                        pending.trigger_cell or pending.target
                    ).as_list(),
                    "direction": pending.direction.value,
                    "observed_head": (
                        None if observed_head is None else observed_head.as_list()
                    ),
                }
            )
        )
        self._pending_escape = None
