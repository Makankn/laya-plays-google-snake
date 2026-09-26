from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .model import Cell, Direction
from .planner import VECTORS
from .policy import build_move_request
from .simulator import SnakeSimulator


@dataclass(frozen=True)
class TrainingSummary:
    episodes: int
    total_steps: int
    total_apples: int
    best_apples: int
    output: Path


def evaluate_laya(
    *,
    model: str,
    episodes: int,
    max_steps: int,
    subfolder: str = "multilingual",
    device: str | None = None,
    seed: int = 10_000,
    validation_samples: int = 200,
    trace: bool = False,
    input_delay_cells: int = 0,
) -> dict[str, object]:
    from .policy import LayaPolicy

    if input_delay_cells not in (0, 1):
        raise ValueError("input_delay_cells must be 0 or 1")
    policy = LayaPolicy(
        model=model,
        subfolder=subfolder,
        device=device,
        warmup=False,
        live_timing=input_delay_cells == 1,
    )
    scores: list[int] = []
    episode_results: list[dict[str, object]] = []
    for episode in range(episodes):
        policy.reset_episode()
        episode_seed = seed + episode
        simulator = SnakeSimulator(seed=episode_seed, max_steps=max_steps)
        state = simulator.reset()
        repeated_steps = 0
        model_decisions = 0
        interventions = 0
        cache_hits = 0
        planning_times: list[float] = []
        intervention_reasons: Counter[str] = Counter()
        recent_moves: deque[dict[str, object]] = deque(maxlen=12)
        queued_turns = 0
        delayed_turns = 0
        pending_escapes: deque[tuple[Cell, Direction, Direction]] = deque()
        pending_turn: Direction | None = None
        stop_reason = "step_limit"
        failure_state = None
        while not simulator.done and state.food is not None:
            policy.observe(state)
            if pending_turn is not None:
                direction = pending_turn
                pending_turn = None
                recent_moves.append({
                    "head": state.head.as_list(),
                    "food": state.food.as_list(),
                    "proposed": None,
                    "executed": direction.value,
                    "reason": "delayed_turn_applied",
                })
            elif pending_escapes:
                trigger, escape, travel = pending_escapes[0]
                direction = escape if state.head == trigger else travel
                if state.head == trigger:
                    queued_turns += 1
                    pending_escapes.popleft()
                recent_moves.append({
                    "head": state.head.as_list(),
                    "food": state.food.as_list(),
                    "proposed": None,
                    "executed": direction.value,
                    "reason": "queued_escape",
                })
            else:
                try:
                    decision = (
                        policy.decide_live(state)
                        if input_delay_cells else policy.decide(state)
                    )
                except (RuntimeError, ValueError) as error:
                    stop_reason = f"policy_error: {error}"
                    failure_state = state.as_dict()
                    break
                model_decisions += decision.intervention_reason not in {
                    "forced_turn_guard", "single_executable_move"
                }
                interventions += decision.intervened
                cache_hits += decision.cache_hit
                if decision.inference_ms > 0:
                    planning_times.append(decision.inference_ms)
                if decision.intervention_reason:
                    intervention_reasons[decision.intervention_reason] += 1
                direction = decision.executed
                recent_moves.append({
                    "head": state.head.as_list(),
                    "food": state.food.as_list(),
                    "proposed": decision.proposed.value,
                    "executed": direction.value,
                    "reason": decision.intervention_reason,
                })
                if input_delay_cells and direction != state.direction:
                    pending_turn = direction
                    direction = state.direction
                    delayed_turns += 1
                    # A queue tied to the proposed path is canceled if the
                    # first key misses its intended cell, as in the live log.
                elif decision.queued_escape is not None:
                    dx, dy = VECTORS[direction]
                    target = Cell(state.head.x + dx, state.head.y + dy)
                    next_target = Cell(target.x + dx, target.y + dy)
                    trigger = next_target if state.food == next_target else target
                    pending_escapes.append((trigger, decision.queued_escape, direction))
                    if decision.following_turn is not None:
                        escape_dx, escape_dy = VECTORS[decision.queued_escape]
                        follow_trigger = Cell(
                            trigger.x + escape_dx, trigger.y + escape_dy
                        )
                        pending_escapes.append((
                            follow_trigger,
                            decision.following_turn,
                            decision.queued_escape,
                        ))
                        if decision.third_turn is not None:
                            follow_dx, follow_dy = VECTORS[decision.following_turn]
                            third_trigger = Cell(
                                follow_trigger.x + follow_dx,
                                follow_trigger.y + follow_dy,
                            )
                            pending_escapes.append((
                                third_trigger,
                                decision.third_turn,
                                decision.following_turn,
                            ))
            result = simulator.step(direction)
            repeated_steps += result.repeated
            if result.state is None:
                stop_reason = "collision"
                break
            state = result.state
        if state.food is None:
            stop_reason = "board_full"
        scores.append(simulator.apples)
        episode_result: dict[str, object] = {
            "seed": episode_seed,
            "apples": simulator.apples,
            "steps": simulator.steps,
            "stop_reason": stop_reason,
            "repeated_steps": repeated_steps,
            "model_decisions": model_decisions,
            "interventions": interventions,
            "cache_hits": cache_hits,
            "median_planning_ms": round(statistics.median(planning_times), 2) if planning_times else 0.0,
            "p90_planning_ms": round(sorted(planning_times)[int(0.9 * (len(planning_times) - 1))], 2) if planning_times else 0.0,
            "max_planning_ms": round(max(planning_times), 2) if planning_times else 0.0,
            "queued_turns": queued_turns,
            "delayed_turns": delayed_turns,
            "intervention_reasons": dict(intervention_reasons),
            "failure_state": failure_state,
        }
        if trace:
            episode_result["last_moves"] = list(recent_moves)
        episode_results.append(episode_result)
    validation = _generate_curriculum(validation_samples, max_steps, seed + 50_000)
    correct = 0
    for sample in validation:
        output = policy.agent.predict(
            sample.prompt,
            {
                "move": {
                    "type": "choice",
                    "instructions": "Choose the best executable move toward food.",
                    "criteria": sample.criteria,
                }
            },
        )
        probabilities = output["answers"]["move"]["probabilities"]
        keys = list(sample.criteria)
        predicted = max(range(len(keys)), key=lambda index: probabilities[keys[index]])
        correct += predicted == sample.action_index
    return {
        "model": model,
        "input_delay_cells": input_delay_cells,
        "episodes": episodes,
        "mean_apples": sum(scores) / len(scores),
        "best_apples": max(scores),
        "worst_apples": min(scores),
        "episode_results": episode_results,
        "move_accuracy": correct / len(validation) if validation else 0.0,
        "validation_samples": len(validation),
    }


@dataclass(frozen=True)
class _Experience:
    prompt: str
    criteria: dict[str, str]
    action_index: int
    reward: float


def train_laya(
    *,
    episodes: int,
    max_steps: int,
    output: str | Path,
    base_model: str = "convaiinnovations/laya",
    subfolder: str = "multilingual",
    device: str | None = None,
    learning_rate: float = 3e-5,
    gamma: float = 0.98,
    entropy_weight: float = 0.01,
    exploration: float = 0.15,
    batch_size: int = 32,
    seed: int = 0,
    imitation_samples: int = 1200,
    imitation_epochs: int = 2,
    emit: Callable[[str], None] = print,
) -> TrainingSummary:
    if episodes < 0 or max_steps < 2 or batch_size < 1:
        raise ValueError("episodes must be non-negative; max_steps and batch_size must be positive")
    if not 0 <= exploration <= 1:
        raise ValueError("exploration must be between 0 and 1")
    try:
        import laya
        import torch
        import torch.nn.functional as functional
        from laya.common import QTYPES, build_sequence, collate_items
    except ImportError as error:
        raise RuntimeError("Install the model runtime with: uv sync --extra laya") from error

    options = {"device": device} if device else {}
    if not Path(base_model).exists() and subfolder:
        options["subfolder"] = subfolder
    agent = laya.load(base_model, **options)
    model = agent.model
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("act_head.")
    ]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed)
    total_steps = total_apples = best_apples = 0

    if imitation_samples > 0 and imitation_epochs > 0:
        curriculum = _generate_curriculum(imitation_samples, max_steps, seed)
        for epoch in range(1, imitation_epochs + 1):
            loss = _supervised_update(
                agent,
                optimizer,
                curriculum,
                batch_size,
                QTYPES,
                build_sequence,
                collate_items,
                functional,
                torch,
            )
            emit(
                json.dumps(
                    {
                        "status": "curriculum",
                        "epoch": epoch,
                        "epochs": imitation_epochs,
                        "samples": len(curriculum),
                        "loss": round(loss, 5),
                    }
                )
            )

    for episode in range(1, episodes + 1):
        simulator = SnakeSimulator(seed=seed + episode, max_steps=max_steps)
        state = simulator.reset()
        trajectory: list[_Experience] = []
        model.eval()
        while not simulator.done and state.food is not None:
            request = build_move_request(state)
            question = {
                "move": {
                    "type": "choice",
                    "instructions": "Choose the best executable move toward food.",
                    "criteria": request.criteria,
                }
            }
            output_payload = agent.predict(request.prompt, question)
            raw = output_payload["answers"]["move"]["probabilities"]
            keys = list(request.criteria)
            weights = [float(raw[key]) for key in keys]
            if rng.random() < exploration:
                action_index = rng.randrange(len(keys))
            else:
                action_index = rng.choices(range(len(keys)), weights=weights, k=1)[0]
            direction = Direction(keys[action_index])
            result = simulator.step(direction)
            trajectory.append(
                _Experience(request.prompt, request.criteria, action_index, result.reward)
            )
            if result.state is None:
                break
            state = result.state

        returns: list[float] = []
        value = 0.0
        for experience in reversed(trajectory):
            value = experience.reward + gamma * value
            returns.append(value)
        returns.reverse()
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum((item - mean) ** 2 for item in returns) / len(returns)
            scale = math.sqrt(variance + 1e-6)
            advantages = [(item - mean) / scale for item in returns]
            _update(
                agent,
                optimizer,
                trajectory,
                advantages,
                batch_size,
                entropy_weight,
                QTYPES,
                build_sequence,
                collate_items,
                functional,
                torch,
            )

        total_steps += simulator.steps
        total_apples += simulator.apples
        best_apples = max(best_apples, simulator.apples)
        if episode == 1 or episode % 5 == 0 or episode == episodes:
            emit(
                json.dumps(
                    {
                        "status": "training",
                        "episode": episode,
                        "episodes": episodes,
                        "steps": simulator.steps,
                        "apples": simulator.apples,
                        "best_apples": best_apples,
                    }
                )
            )

    destination = Path(output)
    _save(agent, destination, episodes, total_steps, total_apples)
    emit(json.dumps({"status": "model_saved", "path": str(destination.resolve())}))
    return TrainingSummary(episodes, total_steps, total_apples, best_apples, destination)


def _update(
    agent,
    optimizer,
    trajectory,
    advantages,
    batch_size,
    entropy_weight,
    qtypes,
    build_sequence,
    collate_items,
    functional,
    torch,
) -> None:
    order = list(range(len(trajectory)))
    random.shuffle(order)
    agent.model.train()
    agent.model.encoder.eval()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        items = []
        selected = []
        weights = []
        for index in indices:
            experience = trajectory[index]
            internal = {
                "t": "choice",
                "ins": "Choose the best executable move toward food.",
                "crit": experience.criteria,
            }
            sequence, markers = build_sequence(
                agent.tok,
                experience.prompt,
                internal,
                agent.cfg.get("max_len", 512),
                agent.cfg.get("head_max_len", 192),
            )
            items.append(
                {"ids": sequence, "markers": markers, "qtype": qtypes["choice"]}
            )
            selected.append(experience.action_index)
            weights.append(advantages[index])
        batch = collate_items([items], agent.tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        use_amp = agent.device.type == "cuda"
        with torch.autocast(
            device_type=agent.device.type, dtype=agent.dtype, enabled=use_amp
        ):
            logits, _ = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
                detach_encoder=True,
            )
            log_probabilities = functional.log_softmax(logits.float(), dim=-1)
            probabilities = log_probabilities.exp()
            row = torch.arange(len(indices), device=agent.device)
            action = torch.tensor(selected, device=agent.device)
            advantage = torch.tensor(weights, device=agent.device, dtype=torch.float32)
            policy_loss = -(log_probabilities[row, action] * advantage).mean()
            entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
            loss = policy_loss - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            1.0,
        )
        optimizer.step()
    agent.model.eval()


def _generate_curriculum(
    sample_count: int, max_steps: int, seed: int
) -> list[_Experience]:
    rng = random.Random(seed + 91_337)
    simulator = SnakeSimulator(seed=seed + 17, max_steps=max_steps)
    state = simulator.reset()
    samples: list[_Experience] = []
    directions = tuple(Direction)
    base_quota, remainder = divmod(sample_count, len(directions))
    quotas = {
        direction: base_quota + (index < remainder)
        for index, direction in enumerate(directions)
    }
    counts = {direction: 0 for direction in directions}
    while len(samples) < sample_count:
        request = build_move_request(state)
        keys = list(request.criteria)
        move_by_direction = {move.direction: move for move in request.moves}
        utilities = []
        for key in keys:
            move = move_by_direction[Direction(key)]
            utility = (
                (20.0 if move.eats else 0.0)
                - (25.0 if move.requires_escape and not move.eats else 0.0)
                - float(move.food_distance)
                + 0.002 * move.reachable_cells
                + 0.001 * move.forward_clearance
            )
            utilities.append(utility)
        target = max(range(len(keys)), key=lambda index: utilities[index])
        target_direction = Direction(keys[target])
        if counts[target_direction] < quotas[target_direction]:
            samples.append(_Experience(request.prompt, request.criteria, target, 0.0))
            counts[target_direction] += 1
        exploratory = Direction(rng.choice(keys))
        result = simulator.step(exploratory)
        if result.state is None or simulator.done or result.state.food is None:
            simulator = SnakeSimulator(
                seed=seed + 17 + len(samples), max_steps=max_steps
            )
            state = simulator.reset()
        else:
            state = result.state
    return samples


def _supervised_update(
    agent,
    optimizer,
    samples,
    batch_size,
    qtypes,
    build_sequence,
    collate_items,
    functional,
    torch,
) -> float:
    order = list(range(len(samples)))
    random.shuffle(order)
    losses: list[float] = []
    agent.model.train()
    agent.model.encoder.eval()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        items = []
        targets = []
        for index in indices:
            sample = samples[index]
            internal = {
                "t": "choice",
                "ins": "Choose the best executable move toward food.",
                "crit": sample.criteria,
            }
            sequence, markers = build_sequence(
                agent.tok,
                sample.prompt,
                internal,
                agent.cfg.get("max_len", 512),
                agent.cfg.get("head_max_len", 192),
            )
            items.append(
                {"ids": sequence, "markers": markers, "qtype": qtypes["choice"]}
            )
            targets.append(sample.action_index)
        batch = collate_items([items], agent.tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=agent.device.type,
            dtype=agent.dtype,
            enabled=agent.device.type == "cuda",
        ):
            logits, _ = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
                detach_encoder=True,
            )
            loss = functional.cross_entropy(
                logits.float(), torch.tensor(targets, device=agent.device)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    agent.model.eval()
    return sum(losses) / len(losses)


def _save(agent, destination: Path, episodes: int, steps: int, apples: int) -> None:
    import torch
    from safetensors.torch import save_file

    destination.mkdir(parents=True, exist_ok=True)
    config = dict(agent.cfg)
    method = (
        "snake-balanced-imitation-frozen-encoder"
        if episodes == 0
        else "snake-balanced-imitation-plus-policy-gradient-frozen-encoder"
    )
    config["training"] = {
        "method": method,
        "episodes": episodes,
        "steps": steps,
        "apples": apples,
    }
    (destination / "rl_agent_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    agent.tok.save_pretrained(destination / "tokenizer")
    agent.model.encoder.config.save_pretrained(destination / "encoder")
    weights = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in agent.model.state_dict().items()
    }
    save_file(weights, destination / "model.safetensors")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
