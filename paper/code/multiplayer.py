from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from .model import Action, DayDecision, Scenario, Weather
from .simulator import SimulationError


@dataclass(frozen=True)
class PlayerState:
    location: int
    cash: Fraction
    water: int
    food: int
    arrival_day: int | None = None
    failed: bool = False


@dataclass(frozen=True)
class MultiDayRecord:
    day: int
    weather: Weather | None
    states: tuple[PlayerState, ...]
    decisions: tuple[DayDecision | None, ...]


@dataclass(frozen=True)
class MultiSimulationResult:
    records: tuple[MultiDayRecord, ...]
    final_values: tuple[Fraction | None, ...]


class JointOnlinePolicy(Protocol):
    def initial_purchases(self, scenario: Scenario) -> tuple[tuple[int, int], ...]: ...

    def decide(
        self,
        scenario: Scenario,
        history: tuple[MultiDayRecord, ...],
        weather: Weather,
    ) -> tuple[DayDecision | None, ...]: ...


def _load(scenario: Scenario, water: int, food: int) -> int:
    return water * scenario.water.mass + food * scenario.food.mass


def _purchase_groups(
    scenario: Scenario,
    states: list[PlayerState],
    decisions: tuple[DayDecision | None, ...],
    before: bool,
) -> list[PlayerState]:
    buyers: dict[int, list[int]] = {}
    for index, (state, decision) in enumerate(zip(states, decisions)):
        if decision is None or state.failed or state.arrival_day is not None:
            continue
        water = decision.buy_before_water if before else decision.buy_water
        food = decision.buy_before_food if before else decision.buy_food
        if water or food:
            if state.location not in scenario.villages:
                raise SimulationError("多人策略试图在非村庄购买")
            buyers.setdefault(state.location, []).append(index)

    updated = list(states)
    for location, indices in buyers.items():
        multiplier = 4 if len(indices) >= 2 else 2
        for index in indices:
            state = updated[index]
            decision = decisions[index]
            assert decision is not None
            water = decision.buy_before_water if before else decision.buy_water
            food = decision.buy_before_food if before else decision.buy_food
            cash = state.cash - multiplier * (
                water * scenario.water.base_price + food * scenario.food.base_price
            )
            new_water = state.water + water
            new_food = state.food + food
            if cash < 0 or _load(scenario, new_water, new_food) > scenario.capacity:
                raise SimulationError("多人村庄采购超过资金或负重")
            updated[index] = PlayerState(
                state.location, cash, new_water, new_food, state.arrival_day, state.failed
            )
    return updated


def simulate_joint_policy(
    scenario: Scenario,
    policy: JointOnlinePolicy,
    weather: tuple[Weather, ...],
) -> MultiSimulationResult:
    n = scenario.player_count
    purchases = policy.initial_purchases(scenario)
    if len(purchases) != n:
        raise SimulationError("初始采购人数与场景不一致")
    states: list[PlayerState] = []
    for water, food in purchases:
        if water < 0 or food < 0 or _load(scenario, water, food) > scenario.capacity:
            raise SimulationError("多人初始采购不合法")
        cash = Fraction(
            scenario.initial_cash
            - water * scenario.water.base_price
            - food * scenario.food.base_price
        )
        if cash < 0:
            raise SimulationError("多人初始采购超过资金")
        states.append(PlayerState(scenario.start, cash, water, food))

    records = [MultiDayRecord(0, None, tuple(states), (None,) * n)]
    for day in range(1, scenario.deadline + 1):
        if all(state.failed or state.arrival_day is not None for state in states):
            break
        today = weather[day - 1]
        decisions = policy.decide(scenario, tuple(records), today)
        if len(decisions) != n:
            raise SimulationError("多人策略行动人数不一致")
        states = _purchase_groups(scenario, states, decisions, before=True)

        walk_groups: dict[tuple[int, int], list[int]] = {}
        mine_groups: dict[int, list[int]] = {}
        for index, (state, decision) in enumerate(zip(states, decisions)):
            if state.failed or state.arrival_day is not None:
                if decision is not None:
                    raise SimulationError("失败或到达玩家仍在行动")
                continue
            if decision is None:
                raise SimulationError("活跃玩家缺少行动")
            if decision.action is Action.WALK:
                if today is Weather.SANDSTORM:
                    raise SimulationError("沙暴日不能行走")
                if decision.destination not in scenario.graph[state.location]:
                    raise SimulationError("多人策略移动不相邻")
                walk_groups.setdefault((state.location, int(decision.destination)), []).append(index)
            elif decision.action is Action.MINE:
                if state.location not in scenario.mines:
                    raise SimulationError("多人策略在非矿山挖矿")
                mine_groups.setdefault(state.location, []).append(index)
            elif decision.action is not Action.STAY:
                raise SimulationError("多人行动类型无效")

        next_states = list(states)
        for index, (state, decision) in enumerate(zip(states, decisions)):
            if decision is None:
                continue
            cash = state.cash
            location = state.location
            if decision.action is Action.WALK:
                group_size = len(walk_groups[(state.location, int(decision.destination))])
                factor = 2 * group_size if group_size >= 2 else 2
                location = int(decision.destination)
            elif decision.action is Action.MINE:
                group_size = len(mine_groups[state.location])
                factor = 3
                cash += Fraction(scenario.mine_income, group_size)
            else:
                factor = 1
            water = state.water - factor * scenario.water.consumption[today]
            food = state.food - factor * scenario.food.consumption[today]
            failed = water < 0 or food < 0 or (
                location != scenario.destination and (water == 0 or food == 0)
            )
            arrival_day = day if location == scenario.destination and not failed else None
            next_states[index] = PlayerState(
                location, cash, water, food, arrival_day, failed
            )
        states = _purchase_groups(scenario, next_states, decisions, before=False)
        records.append(MultiDayRecord(day, today, tuple(states), decisions))

    final_values: list[Fraction | None] = []
    for state in states:
        if state.arrival_day is None or state.failed:
            final_values.append(None)
        else:
            final_values.append(
                state.cash
                + Fraction(scenario.water.base_price * state.water, 2)
                + Fraction(scenario.food.base_price * state.food, 2)
            )
    return MultiSimulationResult(tuple(records), tuple(final_values))
