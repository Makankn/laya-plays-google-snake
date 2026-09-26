import numpy as np

from snake_laya.model import Cell
from snake_laya.parser import ParserConfig, SnakeFrameParser


def test_detects_synthetic_checkerboard_and_objects() -> None:
    image = np.full((190, 230, 3), (75, 120, 45), dtype=np.uint8)
    x0, y0, size, columns, rows = 10, 20, 10, 17, 15
    colors = ((170, 215, 80), (162, 209, 72))
    for y in range(rows):
        for x in range(columns):
            image[y0 + y * size : y0 + (y + 1) * size, x0 + x * size : x0 + (x + 1) * size] = colors[(x + y) % 2]

    for x in (2, 3, 4):
        image[y0 + 7 * size : y0 + 8 * size, x0 + x * size : x0 + (x + 1) * size] = (65, 110, 235)
    image[y0 + 7 * size + 3 : y0 + 7 * size + 6, x0 + 4 * size + 3 : x0 + 4 * size + 6] = (255, 255, 255)
    image[y0 + 7 * size : y0 + 8 * size, x0 + 12 * size : x0 + 13 * size] = (235, 65, 40)

    observation = SnakeFrameParser(ParserConfig(minimum_cell_size_px=8)).parse_array(image)

    assert observation.geometry.columns == 17
    assert observation.geometry.rows == 15
    assert observation.geometry.cell_size_px == 10
    assert observation.snake_cells == frozenset({Cell(2, 7), Cell(3, 7), Cell(4, 7)})
    assert observation.head_hint == Cell(4, 7)
    assert observation.food == Cell(12, 7)
