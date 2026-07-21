from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model import DayDecision, Plan, Scenario, Weather
from .simulator import DailyRecord, SimulationError, SimulationResult, advance_day, simulate


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
    initial = simulate(scenario, Plan(initial_water, initial_food, ()), weather)
    records = [initial.records[0]]
    for day in range(1, scenario.deadline + 1):
        if records[-1].location == scenario.destination:
            break
        decision = policy.decide(scenario, tuple(records), weather[day - 1])
        decisions.append(decision)
        try:
            record = advance_day(scenario, records[-1], decision, weather[day - 1])
        except SimulationError as exc:
            raise SimulationError(f"在线策略在第 {day} 日产生非法决策：{exc}") from exc
        records.append(record)
        if record.location == scenario.destination:
            break
    # 最终仍调用完整模拟器，既生成标准结果对象，也对整条计划再做一次独立入口检查。
    result = simulate(scenario, Plan(initial_water, initial_food, tuple(decisions)), weather)
    return PolicyRun(Plan(initial_water, initial_food, tuple(decisions)), result)
