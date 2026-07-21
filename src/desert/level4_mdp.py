from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .level4_policies import next_step, shortest_distance
from .model import Action, DayDecision, Scenario, Weather
from .weather import IIDWeatherModel


@dataclass(frozen=True)
class BenchmarkOutcome:
    """固定装载、无补给子问题中的总体期望与失败概率。"""

    expected_final_value: float
    failure_probability: float


@dataclass(frozen=True)
class Level4ExactBenchmark:
    """第四关固定装载 MDP 与阈值策略的同口径对照结果。"""

    initial_water: int
    initial_food: int
    sandstorm_budget: int
    optimal: BenchmarkOutcome
    threshold: BenchmarkOutcome
    optimal_states: int
    threshold_states: int

    @property
    def expected_value_gap(self) -> float:
        return self.optimal.expected_final_value - self.threshold.expected_final_value


class Level4FixedLoadMDP:
    """第四关的精确基准：固定初始装载、无村庄补给的天气揭示后 MDP。

    该模型保留完整 5×5 地图、30 天期限、矿山收益以及移动/停留/挖矿
    动作，但将装载固定为给定值。这样状态只含 ``(天数, 位置, 水, 食物)``，
    可精确求解，用于量化阈值规则在相同装载下的动态决策缺口。
    """

    def __init__(
        self,
        scenario: Scenario,
        weather_model: IIDWeatherModel,
        initial_water: int,
        initial_food: int,
        failure_penalty: float = 0.0,
    ) -> None:
        if scenario.level != 4:
            raise ValueError("Level4FixedLoadMDP 仅适用于第四关")
        if scenario.known_weather is not None:
            raise ValueError("精确基准需要未来天气未知的场景")
        if initial_water < 0 or initial_food < 0:
            raise ValueError("初始装载必须非负")
        if initial_water * scenario.water.mass + initial_food * scenario.food.mass > scenario.capacity:
            raise ValueError("初始装载超过负重上限")
        if initial_water * scenario.water.base_price + initial_food * scenario.food.base_price > scenario.initial_cash:
            raise ValueError("初始装载超过初始资金")
        if failure_penalty < 0:
            raise ValueError("失败惩罚系数必须非负")
        self.scenario = scenario
        self.weather_model = weather_model
        self.initial_water = initial_water
        self.initial_food = initial_food
        self.failure_penalty = failure_penalty
        self._optimal_value = lru_cache(maxsize=None)(self._optimal_value_impl)
        self._optimal_after_weather = lru_cache(maxsize=None)(self._optimal_after_weather_impl)
        self._threshold_value = lru_cache(maxsize=None)(self._threshold_value_impl)

    def _salvage(self, water: int, food: int) -> float:
        return 0.5 * self.scenario.water.base_price * water + 0.5 * self.scenario.food.base_price * food

    def _initial_cash(self) -> float:
        return float(
            self.scenario.initial_cash
            - self.initial_water * self.scenario.water.base_price
            - self.initial_food * self.scenario.food.base_price
        )

    def _actions(self, location: int, weather: Weather) -> tuple[DayDecision, ...]:
        actions = [DayDecision(Action.STAY)]
        if weather is not Weather.SANDSTORM:
            actions.extend(DayDecision(Action.WALK, neighbour) for neighbour in sorted(self.scenario.graph[location]))
        if location in self.scenario.mines:
            actions.append(DayDecision(Action.MINE))
        return tuple(actions)

    def _transition(
        self,
        location: int,
        water: int,
        food: int,
        weather: Weather,
        decision: DayDecision,
    ) -> tuple[int, int, int, float] | None:
        if decision.action is Action.WALK:
            if weather is Weather.SANDSTORM or decision.destination not in self.scenario.graph[location]:
                return None
            next_location = int(decision.destination)
            factor, reward = 2, 0.0
        elif decision.action is Action.STAY:
            next_location, factor, reward = location, 1, 0.0
        elif decision.action is Action.MINE:
            if location not in self.scenario.mines:
                return None
            next_location, factor, reward = location, 3, float(self.scenario.mine_income)
        else:
            return None
        next_water = water - factor * self.scenario.water.consumption[weather]
        next_food = food - factor * self.scenario.food.consumption[weather]
        if next_water < 0 or next_food < 0:
            return None
        if next_location != self.scenario.destination and (next_water == 0 or next_food == 0):
            return None
        return next_location, next_water, next_food, reward

    def _terminal_or_future(
        self,
        day: int,
        transition: tuple[int, int, int, float] | None,
        value_function,
        extra_state: tuple[int, int] = (),
    ) -> BenchmarkOutcome:
        if transition is None:
            return BenchmarkOutcome(0.0, 1.0)
        location, water, food, reward = transition
        if location == self.scenario.destination:
            # 初始剩余现金只在成功到达时才计入终值；失败分支必须严格为 0。
            return BenchmarkOutcome(self._initial_cash() + reward + self._salvage(water, food), 0.0)
        if day == self.scenario.deadline:
            return BenchmarkOutcome(0.0, 1.0)
        future = value_function(day + 1, location, water, food, *extra_state)
        return BenchmarkOutcome(reward + future.expected_final_value, future.failure_probability)

    def _optimal_value_impl(self, day: int, location: int, water: int, food: int) -> BenchmarkOutcome:
        expected_value = 0.0
        failure_probability = 0.0
        for weather in self.weather_model.support:
            probability = self.weather_model.probabilities[weather]
            outcome = self._optimal_after_weather(day, location, water, food, weather)
            expected_value += probability * outcome.expected_final_value
            failure_probability += probability * outcome.failure_probability
        return BenchmarkOutcome(expected_value, failure_probability)

    def _optimal_after_weather_impl(
        self,
        day: int,
        location: int,
        water: int,
        food: int,
        weather: Weather,
    ) -> BenchmarkOutcome:
        candidates = [
            self._terminal_or_future(
                day,
                self._transition(location, water, food, weather, decision),
                self._optimal_value,
            )
            for decision in self._actions(location, weather)
        ]
        # 失败终值为零，故没有可行行动时也有一个明确的失败值。
        return max(
            candidates,
            key=lambda item: (
                item.expected_final_value - self.failure_penalty * item.failure_probability,
                -item.failure_probability,
            ),
        )

    def _threshold_decision(
        self,
        day: int,
        location: int,
        water: int,
        food: int,
        storms_seen: int,
        mine_visited: bool,
        weather: Weather,
        sandstorm_budget: int,
    ) -> DayDecision:
        storms_after_reveal = storms_seen + int(weather is Weather.SANDSTORM)
        mine = min(self.scenario.mines)
        distance = shortest_distance(self.scenario, location, self.scenario.destination)
        storms_left = max(0, sandstorm_budget - storms_after_reveal)
        days_after_today = self.scenario.deadline - day
        water_after_mine = water - 3 * self.scenario.water.consumption[weather]
        food_after_mine = food - 3 * self.scenario.food.consumption[weather]
        required_water = 2 * distance * self.scenario.water.consumption[Weather.HOT] + storms_left * self.scenario.water.consumption[Weather.SANDSTORM]
        required_food = 2 * distance * self.scenario.food.consumption[Weather.HOT] + storms_left * self.scenario.food.consumption[Weather.SANDSTORM]
        can_mine = (
            location == mine
            and days_after_today >= distance + storms_left
            and water_after_mine >= required_water
            and food_after_mine >= required_food
        )
        if can_mine:
            return DayDecision(Action.MINE)
        if weather is Weather.SANDSTORM:
            return DayDecision(Action.STAY)
        target = self.scenario.destination if mine_visited else mine
        return DayDecision(Action.WALK, next_step(self.scenario, location, target))

    def _threshold_value_impl(
        self,
        day: int,
        location: int,
        water: int,
        food: int,
        storms_seen: int,
        mine_visited: int,
        sandstorm_budget: int,
    ) -> BenchmarkOutcome:
        expected_value = 0.0
        failure_probability = 0.0
        for weather in self.weather_model.support:
            decision = self._threshold_decision(
                day, location, water, food, storms_seen, bool(mine_visited), weather, sandstorm_budget
            )
            transition = self._transition(location, water, food, weather, decision)
            next_storms = storms_seen + int(weather is Weather.SANDSTORM)
            next_mine_visited = int(bool(mine_visited) or (transition is not None and transition[0] in self.scenario.mines))
            probability = self.weather_model.probabilities[weather]
            outcome = self._terminal_or_future(
                day,
                transition,
                self._threshold_value,
                (next_storms, next_mine_visited, sandstorm_budget),
            )
            expected_value += probability * outcome.expected_final_value
            failure_probability += probability * outcome.failure_probability
        return BenchmarkOutcome(expected_value, failure_probability)

    def solve(self, sandstorm_budget: int) -> Level4ExactBenchmark:
        if sandstorm_budget < 0:
            raise ValueError("沙暴预算必须非负")
        optimal_future = self._optimal_value(1, self.scenario.start, self.initial_water, self.initial_food)
        threshold_future = self._threshold_value(
            1, self.scenario.start, self.initial_water, self.initial_food, 0, 0, sandstorm_budget
        )
        return Level4ExactBenchmark(
            initial_water=self.initial_water,
            initial_food=self.initial_food,
            sandstorm_budget=sandstorm_budget,
            optimal=optimal_future,
            threshold=threshold_future,
            optimal_states=self._optimal_value.cache_info().currsize,
            threshold_states=self._threshold_value.cache_info().currsize,
        )
