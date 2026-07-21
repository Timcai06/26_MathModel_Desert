from __future__ import annotations

from .model import Action, DayDecision, Plan


def _walk(destination: int, water: int = 0, food: int = 0) -> DayDecision:
    return DayDecision(Action.WALK, destination, water, food)


def _stay(water: int = 0, food: int = 0) -> DayDecision:
    return DayDecision(Action.STAY, buy_water=water, buy_food=food)


def _mine() -> DayDecision:
    return DayDecision(Action.MINE)


LEVEL_1_REFERENCE = Plan(
    178,
    333,
    (
        _walk(25), _walk(24), _walk(23), _stay(), _walk(22), _walk(9), _stay(),
        _walk(15, water=163), _walk(13), _walk(12), _stay(), _mine(), _mine(),
        _mine(), _mine(), _mine(), _stay(), _mine(), _mine(), _walk(13),
        _walk(15, water=36, food=16), _walk(9), _walk(21), _walk(27),
    ),
)


LEVEL_2_REFERENCE = Plan(
    130,
    405,
    (
        _walk(2), _walk(3), _walk(4), _stay(), _walk(5), _walk(13), _stay(),
        _walk(22), _walk(30), _walk(39, water=10), _stay(water=179), _walk(30),
        _mine(), _mine(), _mine(), _mine(), _mine(), _mine(),
        _walk(39, water=196, food=86), _walk(46), _walk(55),
        _mine(), _mine(), _mine(), _mine(), _mine(), _mine(), _mine(),
        _walk(56), _walk(64),
    ),
)
