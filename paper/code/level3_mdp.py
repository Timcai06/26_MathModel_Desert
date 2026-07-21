from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf

from .model import Action, DayDecision, Scenario, Weather
from .online import OnlinePolicy
from .simulator import DailyRecord
from .weather import IIDWeatherModel


NEGATIVE_INFINITY = -inf


@dataclass(frozen=True)
class Level3MDPSolution:
    policy: "Level3MDPPolicy"
    expected_final_value: float
    initial_water: int
    initial_food: int
    states_evaluated: int


class Level3MDPSolver:
    """无村庄、当前天气可见时的有限时域精确动态规划。"""

    def __init__(self, scenario: Scenario, weather_model: IIDWeatherModel):
        if scenario.villages:
            raise ValueError("Level3MDPSolver 仅适用于无村庄关卡")
        if scenario.known_weather is not None:
            raise ValueError("该求解器用于未来天气未知的关卡")
        self.scenario = scenario
        self.weather_model = weather_model
        self._cached_value = lru_cache(maxsize=None)(self._value_impl)

    def _salvage(self, water: int, food: int) -> float:
        return 0.5 * self.scenario.water.base_price * water + 0.5 * self.scenario.food.base_price * food

    def _actions(self, location: int, weather: Weather) -> tuple[DayDecision, ...]:
        actions: list[DayDecision] = []
        if weather is not Weather.SANDSTORM:
            actions.extend(DayDecision(Action.WALK, neighbour) for neighbour in sorted(self.scenario.graph[location]))
        if location in self.scenario.mines:
            actions.append(DayDecision(Action.MINE))
        actions.append(DayDecision(Action.STAY))
        return tuple(actions)

    def _transition(
        self,
        location: int,
        water: int,
        food: int,
        weather: Weather,
        decision: DayDecision,
    ) -> tuple[int, int, int, int] | None:
        if decision.action is Action.WALK:
            if weather is Weather.SANDSTORM or decision.destination not in self.scenario.graph[location]:
                return None
            next_location = int(decision.destination)
            factor = 2
            reward = 0
        elif decision.action is Action.MINE:
            if location not in self.scenario.mines:
                return None
            next_location = location
            factor = 3
            reward = self.scenario.mine_income
        else:
            next_location = location
            factor = 1
            reward = 0
        next_water = water - factor * self.scenario.water.consumption[weather]
        next_food = food - factor * self.scenario.food.consumption[weather]
        if next_water < 0 or next_food < 0:
            return None
        if next_location != self.scenario.destination and (next_water == 0 or next_food == 0):
            return None
        return next_location, next_water, next_food, reward

    def value(self, day: int, location: int, water: int, food: int) -> float:
        return self._cached_value(day, location, water, food)

    def _value_impl(self, day: int, location: int, water: int, food: int) -> float:
        """当第 ``day`` 日天气尚未揭示时的最优未来收益（不含当前现金）。"""
        if location == self.scenario.destination:
            return self._salvage(water, food)
        if day > self.scenario.deadline:
            return NEGATIVE_INFINITY
        expected = 0.0
        for weather in self.weather_model.support:
            probability = self.weather_model.probabilities[weather]
            best, _ = self.best_after_weather(day, location, water, food, weather)
            if best == NEGATIVE_INFINITY:
                return NEGATIVE_INFINITY
            expected += probability * best
        return expected

    def best_after_weather(
        self,
        day: int,
        location: int,
        water: int,
        food: int,
        weather: Weather,
    ) -> tuple[float, DayDecision | None]:
        best_value = NEGATIVE_INFINITY
        best_decision: DayDecision | None = None
        for decision in self._actions(location, weather):
            transition = self._transition(location, water, food, weather, decision)
            if transition is None:
                continue
            next_location, next_water, next_food, reward = transition
            if next_location == self.scenario.destination:
                candidate = reward + self._salvage(next_water, next_food)
            elif day == self.scenario.deadline:
                candidate = NEGATIVE_INFINITY
            else:
                future = self.value(day + 1, next_location, next_water, next_food)
                candidate = NEGATIVE_INFINITY if future == NEGATIVE_INFINITY else reward + future
            if candidate > best_value + 1e-9:
                best_value = candidate
                best_decision = decision
        return best_value, best_decision

    def solve(self) -> Level3MDPSolution:
        max_water = min(
            self.scenario.capacity // self.scenario.water.mass,
            self.scenario.deadline * 3 * max(self.scenario.water.consumption.values()),
        )
        max_food = min(
            self.scenario.capacity // self.scenario.food.mass,
            self.scenario.deadline * 3 * max(self.scenario.food.consumption.values()),
        )
        best_value = NEGATIVE_INFINITY
        best_water = best_food = 0
        for water in range(1, max_water + 1):
            max_food_by_load = (self.scenario.capacity - water * self.scenario.water.mass) // self.scenario.food.mass
            for food in range(1, min(max_food, max_food_by_load) + 1):
                cash = self.scenario.initial_cash - (
                    water * self.scenario.water.base_price + food * self.scenario.food.base_price
                )
                if cash < 0:
                    continue
                future = self.value(1, self.scenario.start, water, food)
                if future == NEGATIVE_INFINITY:
                    continue
                objective = cash + future
                if objective > best_value + 1e-9:
                    best_value = objective
                    best_water, best_food = water, food
        if best_value == NEGATIVE_INFINITY:
            raise RuntimeError("未找到对天气支撑集稳健可行的策略")
        policy = Level3MDPPolicy(self, best_water, best_food)
        return Level3MDPSolution(
            policy,
            best_value,
            best_water,
            best_food,
            self._cached_value.cache_info().currsize,
        )


@dataclass(frozen=True)
class Level3MDPPolicy(OnlinePolicy):
    solver: Level3MDPSolver
    water: int
    food: int

    def initial_purchase(self, scenario: Scenario) -> tuple[int, int]:
        return self.water, self.food

    def decide(
        self,
        scenario: Scenario,
        history: tuple[DailyRecord, ...],
        weather: Weather,
    ) -> DayDecision:
        current = history[-1]
        day = current.day + 1
        _, decision = self.solver.best_after_weather(
            day, current.location, current.water, current.food, weather
        )
        if decision is None:
            raise RuntimeError(f"第 {day} 日状态无可行行动")
        return decision
