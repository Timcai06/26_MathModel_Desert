from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import Action, DayDecision, Scenario, Weather
from .online import OnlinePolicy
from .simulator import DailyRecord


def shortest_distance(scenario: Scenario, start: int, destination: int) -> int:
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, distance = queue.popleft()
        if node == destination:
            return distance
        for neighbour in scenario.graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    raise ValueError(f"{start} 与 {destination} 不连通")


def next_step(scenario: Scenario, start: int, destination: int) -> int:
    if start == destination:
        return start
    target_distance = shortest_distance(scenario, start, destination) - 1
    candidates = [
        neighbour
        for neighbour in scenario.graph[start]
        if shortest_distance(scenario, neighbour, destination) == target_distance
    ]
    return min(candidates)


@dataclass(frozen=True)
class DirectRobustPolicy(OnlinePolicy):
    sandstorm_budget: int

    def initial_purchase(self, scenario: Scenario) -> tuple[int, int]:
        distance = shortest_distance(scenario, scenario.start, scenario.destination)
        walk_water = 2 * distance * max(
            scenario.water.consumption[Weather.SUNNY], scenario.water.consumption[Weather.HOT]
        )
        walk_food = 2 * distance * max(
            scenario.food.consumption[Weather.SUNNY], scenario.food.consumption[Weather.HOT]
        )
        return (
            walk_water + self.sandstorm_budget * scenario.water.consumption[Weather.SANDSTORM],
            walk_food + self.sandstorm_budget * scenario.food.consumption[Weather.SANDSTORM],
        )

    def decide(
        self,
        scenario: Scenario,
        history: tuple[DailyRecord, ...],
        weather: Weather,
    ) -> DayDecision:
        current = history[-1]
        if weather is Weather.SANDSTORM:
            return DayDecision(Action.STAY)
        return DayDecision(Action.WALK, next_step(scenario, current.location, scenario.destination))


@dataclass(frozen=True)
class MineThresholdPolicy(OnlinePolicy):
    sandstorm_budget: int
    initial_water: int = 240
    initial_food: int = 240

    def initial_purchase(self, scenario: Scenario) -> tuple[int, int]:
        return self.initial_water, self.initial_food

    def _observed_storms(self, history: tuple[DailyRecord, ...], today: Weather) -> int:
        previous = sum(record.weather is Weather.SANDSTORM for record in history if record.weather)
        return previous + int(today is Weather.SANDSTORM)

    def _can_mine(
        self,
        scenario: Scenario,
        current: DailyRecord,
        weather: Weather,
        history: tuple[DailyRecord, ...],
    ) -> bool:
        distance = shortest_distance(scenario, current.location, scenario.destination)
        storms_left = max(0, self.sandstorm_budget - self._observed_storms(history, weather))
        days_after_today = scenario.deadline - (current.day + 1)
        if days_after_today < distance + storms_left:
            return False
        water_after = current.water - 3 * scenario.water.consumption[weather]
        food_after = current.food - 3 * scenario.food.consumption[weather]
        required_water = (
            2 * distance * scenario.water.consumption[Weather.HOT]
            + storms_left * scenario.water.consumption[Weather.SANDSTORM]
        )
        required_food = (
            2 * distance * scenario.food.consumption[Weather.HOT]
            + storms_left * scenario.food.consumption[Weather.SANDSTORM]
        )
        return water_after >= required_water and food_after >= required_food

    def decide(
        self,
        scenario: Scenario,
        history: tuple[DailyRecord, ...],
        weather: Weather,
    ) -> DayDecision:
        current = history[-1]
        mine = min(scenario.mines)
        visited_mine = any(record.location == mine for record in history)
        if current.location == mine and self._can_mine(scenario, current, weather, history):
            return DayDecision(Action.MINE)
        if weather is Weather.SANDSTORM:
            return DayDecision(Action.STAY)
        target = scenario.destination if visited_mine else mine
        return DayDecision(Action.WALK, next_step(scenario, current.location, target))

