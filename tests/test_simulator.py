from collections import Counter

from snake_laya.model import Cell, Direction, SnakeState
from snake_laya.simulator import SnakeSimulator
from snake_laya.training import _generate_curriculum


def test_simulator_rewards_food_and_grows() -> None:
    simulator = SnakeSimulator(seed=1)
    state = simulator.reset()
    simulator.state = SnakeState(
        state.geometry,
        (Cell(4, 7), Cell(3, 7), Cell(2, 7)),
        Cell(5, 7),
        Direction.RIGHT,
        0,
    )

    result = simulator.step(Direction.RIGHT)

    assert result.ate
    assert result.reward > 9
    assert result.state is not None
    assert len(result.state.snake) == 4


def test_simulator_penalizes_wall_collision() -> None:
    simulator = SnakeSimulator(seed=1)
    state = simulator.reset()
    simulator.state = SnakeState(
        state.geometry,
        (Cell(16, 7), Cell(15, 7), Cell(14, 7)),
        Cell(4, 4),
        Direction.RIGHT,
        0,
    )

    result = simulator.step(Direction.RIGHT)

    assert result.dead
    assert result.reward == -10
    assert simulator.done


def test_simulator_treats_current_tail_as_occupied() -> None:
    simulator = SnakeSimulator(seed=1)
    state = simulator.reset()
    simulator.state = SnakeState(
        state.geometry,
        (Cell(15, 8), Cell(16, 8), Cell(16, 7), Cell(15, 7)),
        Cell(16, 3),
        Direction.LEFT,
        0,
    )

    result = simulator.step(Direction.UP)

    assert result.dead
    assert result.reward == -10


def test_curriculum_balances_target_directions() -> None:
    samples = _generate_curriculum(40, 80, 7)

    targets = Counter(tuple(sample.criteria)[sample.action_index] for sample in samples)

    assert targets == Counter({direction.value: 10 for direction in Direction})
    progress_labels = ("toward food", "away from food", "same food distance", "eat food now")
    assert all(
        any(label in description for label in progress_labels)
        for sample in samples
        for description in sample.criteria.values()
    )
