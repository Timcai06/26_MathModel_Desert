from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Weather(str, Enum):
    SUNNY = "晴朗"
    HOT = "高温"
    SANDSTORM = "沙暴"


class Action(str, Enum):
    WALK = "行走"
    STAY = "停留"
    MINE = "挖矿"


@dataclass(frozen=True)
class ResourceSpec:
    mass: int
    base_price: int
    consumption: Mapping[Weather, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumption", MappingProxyType(dict(self.consumption)))


@dataclass(frozen=True)
class Scenario:
    level: int
    node_count: int
    graph: Mapping[int, frozenset[int]]
    start: int
    destination: int
    villages: frozenset[int]
    mines: frozenset[int]
    deadline: int
    capacity: int
    initial_cash: int
    mine_income: int
    water: ResourceSpec
    food: ResourceSpec
    known_weather: tuple[Weather, ...] | None
    weather_horizon: int
    player_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph", MappingProxyType(dict(self.graph)))


@dataclass(frozen=True)
class DayDecision:
    action: Action
    destination: int | None = None
    buy_water: int = 0
    buy_food: int = 0
    buy_before_water: int = 0
    buy_before_food: int = 0


@dataclass(frozen=True)
class Plan:
    initial_water: int
    initial_food: int
    decisions: tuple[DayDecision, ...]
