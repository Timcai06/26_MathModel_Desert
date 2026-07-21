from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model import DayDecision, Plan, Scenario, Weather
from .simulator import DailyRecord, SimulationError, SimulationResult, simulate


class OnlinePolicy(Protocol):
    def initial_purchase(self, scenario: Scenario) -> tuple[int, int]: ...

    def decide(
        self,
        scenario: Scenario,
        history: tuple[DailyRecord, ...],
        weather: Weather,
    ) -> DayDecision: ...


@dataclass(frozen=True)
class PolicyRun:
    plan: Plan
    result: SimulationResult


def simulate_policy(
    scenario: Scenario,
    policy: OnlinePolicy,
    weather: tuple[Weather, ...],
) -> PolicyRun:
    """按“当天天气先揭示、随后决策”的时序执行反馈策略。"""
    initial_water, initial_food = policy.initial_purchase(scenario)
    decisions: list[DayDecision] = []
    for day in range(1, scenario.deadline + 1):
        partial = simulate(scenario, Plan(initial_water, initial_food, tuple(decisions)), weather)
        if partial.reached_destination:
            break
        decision = policy.decide(scenario, partial.records, weather[day - 1])
        decisions.append(decision)
        try:
            updated = simulate(scenario, Plan(initial_water, initial_food, tuple(decisions)), weather)
        except SimulationError as exc:
            raise SimulationError(f"在线策略在第 {day} 日产生非法决策：{exc}") from exc
        if updated.reached_destination:
            break
    result = simulate(scenario, Plan(initial_water, initial_food, tuple(decisions)), weather)
    return PolicyRun(Plan(initial_water, initial_food, tuple(decisions)), result)

