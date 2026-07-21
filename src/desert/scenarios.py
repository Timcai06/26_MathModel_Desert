from __future__ import annotations

from .model import ResourceSpec, Scenario, Weather


LEVEL_12_WEATHER = tuple(
    Weather(x)
    for x in (
        "高温", "高温", "晴朗", "沙暴", "晴朗", "高温", "沙暴", "晴朗", "高温", "高温",
        "沙暴", "高温", "晴朗", "高温", "高温", "高温", "沙暴", "沙暴", "高温", "高温",
        "晴朗", "晴朗", "高温", "晴朗", "沙暴", "高温", "晴朗", "晴朗", "高温", "高温",
    )
)

LEVEL_5_WEATHER = tuple(
    Weather(x)
    for x in ("晴朗", "高温", "晴朗", "晴朗", "晴朗", "晴朗", "高温", "高温", "高温", "高温")
)


def _undirected(raw: dict[int, set[int]], node_count: int) -> dict[int, frozenset[int]]:
    graph = {i: set(raw.get(i, set())) for i in range(1, node_count + 1)}
    for node, neighbours in tuple(graph.items()):
        for neighbour in neighbours:
            if not 1 <= neighbour <= node_count or neighbour == node:
                raise ValueError(f"非法边 {node}-{neighbour}")
            graph[neighbour].add(node)
    return {node: frozenset(neighbours) for node, neighbours in graph.items()}


LEVEL_1_GRAPH = _undirected(
    {
        1: {2, 25}, 2: {1, 3}, 3: {2, 4, 25}, 4: {3, 5, 24, 25},
        5: {4, 6, 24}, 6: {5, 7, 23, 24}, 7: {6, 8, 22}, 8: {7, 9, 22},
        9: {8, 10, 15, 16, 17, 21, 22}, 10: {9, 11, 13, 15},
        11: {10, 12, 13}, 12: {11, 13, 14}, 13: {10, 11, 12, 14, 15},
        14: {12, 13, 15, 16}, 15: {9, 10, 13, 14, 16},
        16: {9, 14, 15, 17, 18}, 17: {9, 16, 18, 21},
        18: {16, 17, 19, 20}, 19: {18, 20}, 20: {18, 19, 21},
        21: {9, 17, 20, 22, 23, 27}, 22: {7, 8, 9, 21, 23},
        23: {6, 21, 22, 24, 26}, 24: {4, 5, 6, 23, 25, 26},
        25: {1, 3, 4, 24, 26}, 26: {23, 24, 25, 27}, 27: {21, 26},
    },
    27,
)


def _hex_8_by_8() -> dict[int, frozenset[int]]:
    raw: dict[int, set[int]] = {i: set() for i in range(1, 65)}

    def node(row: int, col: int) -> int:
        return (row - 1) * 8 + col

    for row in range(1, 9):
        for col in range(1, 9):
            here = node(row, col)
            if col < 8:
                raw[here].add(node(row, col + 1))
            if row < 8:
                below_cols = (col - 1, col) if row % 2 else (col, col + 1)
                for below_col in below_cols:
                    if 1 <= below_col <= 8:
                        raw[here].add(node(row + 1, below_col))
    return _undirected(raw, 64)


LEVEL_2_GRAPH = _hex_8_by_8()

LEVEL_35_GRAPH = _undirected(
    {
        1: {2, 4, 5}, 2: {1, 3, 4}, 3: {2, 4, 8, 9},
        4: {1, 2, 3, 5, 6, 7}, 5: {1, 4, 6}, 6: {4, 5, 7, 12, 13},
        7: {4, 6, 11, 12}, 8: {3, 9}, 9: {3, 8, 10, 11},
        10: {9, 11, 13}, 11: {7, 9, 10, 12, 13},
        12: {6, 7, 11, 13}, 13: {6, 10, 11, 12},
    },
    13,
)


def _square_5_by_5() -> dict[int, frozenset[int]]:
    raw: dict[int, set[int]] = {i: set() for i in range(1, 26)}
    for row in range(5):
        for col in range(5):
            here = row * 5 + col + 1
            if col < 4:
                raw[here].add(here + 1)
            if row < 4:
                raw[here].add(here + 5)
    return _undirected(raw, 25)


LEVEL_46_GRAPH = _square_5_by_5()

WATER_12 = ResourceSpec(3, 5, {Weather.SUNNY: 5, Weather.HOT: 8, Weather.SANDSTORM: 10})
FOOD_12 = ResourceSpec(2, 10, {Weather.SUNNY: 7, Weather.HOT: 6, Weather.SANDSTORM: 10})
WATER_36 = ResourceSpec(3, 5, {Weather.SUNNY: 3, Weather.HOT: 9, Weather.SANDSTORM: 10})
FOOD_36 = ResourceSpec(2, 10, {Weather.SUNNY: 4, Weather.HOT: 9, Weather.SANDSTORM: 10})


def _scenario(
    level: int,
    graph: dict[int, frozenset[int]],
    start: int,
    destination: int,
    villages: set[int],
    mines: set[int],
    deadline: int,
    mine_income: int,
    water: ResourceSpec,
    food: ResourceSpec,
    weather: tuple[Weather, ...] | None,
    horizon: int,
    players: int = 1,
) -> Scenario:
    return Scenario(
        level=level,
        node_count=len(graph),
        graph=graph,
        start=start,
        destination=destination,
        villages=frozenset(villages),
        mines=frozenset(mines),
        deadline=deadline,
        capacity=1200,
        initial_cash=10000,
        mine_income=mine_income,
        water=water,
        food=food,
        known_weather=weather,
        weather_horizon=horizon,
        player_count=players,
    )


SCENARIOS = {
    1: _scenario(1, LEVEL_1_GRAPH, 1, 27, {15}, {12}, 30, 1000, WATER_12, FOOD_12, LEVEL_12_WEATHER, 30),
    2: _scenario(2, LEVEL_2_GRAPH, 1, 64, {39, 62}, {30, 55}, 30, 1000, WATER_12, FOOD_12, LEVEL_12_WEATHER, 30),
    3: _scenario(3, LEVEL_35_GRAPH, 1, 13, set(), {9}, 10, 200, WATER_36, FOOD_36, None, 1),
    4: _scenario(4, LEVEL_46_GRAPH, 1, 25, {14}, {18}, 30, 1000, WATER_36, FOOD_36, None, 1),
    5: _scenario(5, LEVEL_35_GRAPH, 1, 13, set(), {9}, 10, 200, WATER_36, FOOD_36, LEVEL_5_WEATHER, 10, 2),
    6: _scenario(6, LEVEL_46_GRAPH, 1, 25, {14}, {18}, 30, 1000, WATER_36, FOOD_36, None, 1, 3),
}


def get_scenario(level: int) -> Scenario:
    try:
        return SCENARIOS[level]
    except KeyError as exc:
        raise ValueError(f"不存在第 {level} 关") from exc

