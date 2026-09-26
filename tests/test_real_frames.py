from pathlib import Path

import pytest

from snake_laya.model import Cell
from snake_laya.parser import SnakeFrameParser


FAILURE_FRAME = Path("captures/desync-20260922-004336-008567.png")


@pytest.mark.skipif(not FAILURE_FRAME.is_file(), reason="local gameplay corpus unavailable")
def test_failure_frame_uses_visible_eyes_for_head() -> None:
    observation = SnakeFrameParser().parse_file(FAILURE_FRAME)

    assert observation.geometry.columns == 17
    assert observation.geometry.rows == 15
    assert observation.head_hint == Cell(4, 11)
    assert observation.head_confidence >= 0.55
