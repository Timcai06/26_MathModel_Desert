from __future__ import annotations

from dataclasses import dataclass

from .model import Action, DayDecision, Plan, Scenario, Weather


class SimulationError(ValueError):
    """策略违反题面规则。"""


@dataclass(frozen=True)
class DailyRecord:
    day: int
    location: int
    cash: int
    water: int
    food: int
    weather: Weather | None
    action: Action | None
    buy_water: int = 0
    buy_food: int = 0
    buy_before_water: int = 0
    buy_before_food: int = 0


@dataclass(frozen=True)
class SimulationResult:
    scenario_level: int
    records: tuple[DailyRecord, ...]
    reached_destination: bool
    arrival_day: int | None
    final_value: float | None


def _load(scenario: Scenario, water: int, food: int) -> int:
    return water * scenario.water.mass + food * scenario.food.mass


def _purchase_cost(scenario: Scenario, water: int, food: int, multiplier: int) -> int:
    return multiplier * (water * scenario.water.base_price + food * scenario.food.base_price)


def _validate_purchase_quantities(water: int, food: int) -> None:
    if not isinstance(water, int) or not isinstance(food, int) or water < 0 or food < 0:
        raise SimulationError("水和食物必须按非负整数箱购买")


def simulate(
    scenario: Scenario,
    plan: Plan,
    weather: tuple[Weather, ...] | None = None,
) -> SimulationResult:
    """逐日执行单人策略。

    ``buy_before_*`` 在当日天气揭示后、行动前购买；``buy_*`` 在当日行动和
    消耗后购买。两者都只能发生在村庄。
    第 0 日在起点按基准价一次性购买初始物资。
    """
    actual_weather = weather if weather is not None else scenario.known_weather
    if actual_weather is None:
        raise SimulationError("本关天气未预先给定，必须显式传入实际天气序列")
    if len(actual_weather) < min(len(plan.decisions), scenario.deadline):
        raise SimulationError("天气序列短于策略执行天数")

    _validate_purchase_quantities(plan.initial_water, plan.initial_food)
    cash = scenario.initial_cash - _purchase_cost(
        scenario, plan.initial_water, plan.initial_food, multiplier=1
    )
    water = plan.initial_water
    food = plan.initial_food
    location = scenario.start
    if cash < 0:
        raise SimulationError("第 0 日初始采购超过初始资金")
    if _load(scenario, water, food) > scenario.capacity:
        raise SimulationError("第 0 日初始物资超过负重上限")

    records = [DailyRecord(0, location, cash, water, food, None, None)]
    arrival_day: int | None = None

    for day, decision in enumerate(plan.decisions, start=1):
        if day > scenario.deadline:
            raise SimulationError("策略超过截止日期")
        if arrival_day is not None:
            raise SimulationError("到达终点后不能继续行动")
        _validate_purchase_quantities(decision.buy_water, decision.buy_food)
        _validate_purchase_quantities(decision.buy_before_water, decision.buy_before_food)
        today = actual_weather[day - 1]

        if decision.buy_before_water or decision.buy_before_food:
            if location not in scenario.villages:
                raise SimulationError(f"第 {day} 日行动前不在村庄，不能补给")
            cash -= _purchase_cost(
                scenario, decision.buy_before_water, decision.buy_before_food, multiplier=2
            )
            water += decision.buy_before_water
            food += decision.buy_before_food
            if cash < 0:
                raise SimulationError(f"第 {day} 日行动前采购超过可用资金")
            if _load(scenario, water, food) > scenario.capacity:
                raise SimulationError(f"第 {day} 日行动前采购后超过负重上限")

        if decision.action is Action.WALK:
            if today is Weather.SANDSTORM:
                raise SimulationError(f"第 {day} 日沙暴天气不能行走")
            if decision.destination is None or decision.destination not in scenario.graph[location]:
                raise SimulationError(f"第 {day} 日移动目的地与区域 {location} 不相邻")
            location = decision.destination
            consumption_factor = 2
        elif decision.action is Action.STAY:
            if decision.destination is not None:
                raise SimulationError(f"第 {day} 日停留时不应填写移动目的地")
            consumption_factor = 1
        elif decision.action is Action.MINE:
            if decision.destination is not None:
                raise SimulationError(f"第 {day} 日挖矿时不应填写移动目的地")
            if location not in scenario.mines:
                raise SimulationError(f"第 {day} 日所在区域 {location} 不是矿山")
            consumption_factor = 3
            cash += scenario.mine_income
        else:
            raise SimulationError(f"第 {day} 日行动类型无效")

        water -= scenario.water.consumption[today] * consumption_factor
        food -= scenario.food.consumption[today] * consumption_factor
        if water < 0 or food < 0:
            raise SimulationError(f"第 {day} 日行动途中物资不足")

        if decision.buy_water or decision.buy_food:
            if location not in scenario.villages:
                raise SimulationError(f"第 {day} 日不在村庄，不能补给")
            cash -= _purchase_cost(
                scenario, decision.buy_water, decision.buy_food, multiplier=2
            )
            water += decision.buy_water
            food += decision.buy_food
            if cash < 0:
                raise SimulationError(f"第 {day} 日村庄采购超过可用资金")
            if _load(scenario, water, food) > scenario.capacity:
                raise SimulationError(f"第 {day} 日村庄采购后超过负重上限")

        if location != scenario.destination and (water <= 0 or food <= 0):
            raise SimulationError(f"第 {day} 日结束时物资耗尽且尚未到达终点")

        records.append(
            DailyRecord(
                day, location, cash, water, food, today, decision.action,
                decision.buy_water, decision.buy_food,
                decision.buy_before_water, decision.buy_before_food,
            )
        )
        if location == scenario.destination:
            arrival_day = day

    reached = arrival_day is not None
    final_value = None
    if reached:
        final_value = (
            cash
            + 0.5 * scenario.water.base_price * water
            + 0.5 * scenario.food.base_price * food
        )
    return SimulationResult(scenario.level, tuple(records), reached, arrival_day, final_value)
