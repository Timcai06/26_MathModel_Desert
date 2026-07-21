from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .level4_policies import shortest_distance
from .model import Action, DayDecision, Scenario, Weather


@dataclass(frozen=True)
class ActionPath:
    decisions: tuple[DayDecision, ...]


@dataclass(frozen=True)
class PlanEvaluation:
    path: ActionPath
    initial_water: int
    initial_food: int
    arrival_day: int
    mining_revenue: Fraction
    final_value: Fraction


@dataclass(frozen=True)
class Level5Equilibrium:
    player1: PlanEvaluation
    player2: PlanEvaluation
    player1_deviation_gain: Fraction
    player2_deviation_gain: Fraction


def _schedule(scenario: Scenario, path: ActionPath) -> tuple[tuple[int, DayDecision], ...]:
    location = scenario.start
    schedule: list[tuple[int, DayDecision]] = []
    for day, decision in enumerate(path.decisions, start=1):
        if location == scenario.destination:
            raise ValueError("到达终点后仍有行动")
        if decision.action is Action.WALK:
            if decision.destination not in scenario.graph[location]:
                raise ValueError(f"第 {day} 天移动不相邻")
            next_location = int(decision.destination)
        elif decision.action is Action.MINE:
            if location not in scenario.mines:
                raise ValueError(f"第 {day} 天不在矿山")
            next_location = location
        elif decision.action is Action.STAY:
            next_location = location
        else:
            raise ValueError("未知行动")
        schedule.append((location, decision))
        location = next_location
    if location != scenario.destination:
        raise ValueError("行动序列未到达终点")
    return tuple(schedule)


def _daily_effect(
    scenario: Scenario,
    weather: Weather,
    own_origin: int,
    own: DayDecision,
    opponent_day: tuple[int, DayDecision] | None,
) -> tuple[int, int, Fraction]:
    revenue = Fraction(0)
    if own.action is Action.WALK:
        same_edge = (
            opponent_day is not None
            and opponent_day[1].action is Action.WALK
            and opponent_day[0] == own_origin
            and opponent_day[1].destination == own.destination
        )
        factor = 4 if same_edge else 2
    elif own.action is Action.MINE:
        shared_mine = (
            opponent_day is not None
            and opponent_day[1].action is Action.MINE
            and opponent_day[0] == own_origin
        )
        factor = 3
        revenue = Fraction(scenario.mine_income, 2 if shared_mine else 1)
    else:
        factor = 1
    return (
        factor * scenario.water.consumption[weather],
        factor * scenario.food.consumption[weather],
        revenue,
    )


def evaluate_path(
    scenario: Scenario,
    own_path: ActionPath,
    opponent_path: ActionPath | None,
) -> PlanEvaluation:
    if scenario.known_weather is None:
        raise ValueError("第五关必须给定完整天气")
    own_schedule = _schedule(scenario, own_path)
    opponent_schedule = _schedule(scenario, opponent_path) if opponent_path else ()
    water = food = 0
    revenue = Fraction(0)
    for day, (origin, decision) in enumerate(own_schedule, start=1):
        opponent_day = opponent_schedule[day - 1] if day <= len(opponent_schedule) else None
        dw, df, daily_revenue = _daily_effect(
            scenario, scenario.known_weather[day - 1], origin, decision, opponent_day
        )
        water += dw
        food += df
        revenue += daily_revenue
    if water * scenario.water.mass + food * scenario.food.mass > scenario.capacity:
        raise ValueError("行动序列所需初始物资超过负重")
    purchase_cost = water * scenario.water.base_price + food * scenario.food.base_price
    if purchase_cost > scenario.initial_cash:
        raise ValueError("行动序列所需初始物资超过初始资金")
    return PlanEvaluation(
        own_path,
        water,
        food,
        len(own_schedule),
        revenue,
        Fraction(scenario.initial_cash - purchase_cost) + revenue,
    )


def best_response(scenario: Scenario, opponent_path: ActionPath | None) -> PlanEvaluation:
    """对固定对手计划做完整有限时域动态规划，返回精确最佳响应。"""
    if scenario.known_weather is None:
        raise ValueError("第五关必须给定完整天气")
    opponent_schedule = _schedule(scenario, opponent_path) if opponent_path else ()
    # 状态值记录达到(day, location, water, food)时可获得的最大挖矿收入及路径。
    states: dict[tuple[int, int, int], tuple[Fraction, tuple[DayDecision, ...]]] = {
        (scenario.start, 0, 0): (Fraction(0), ())
    }
    terminals: list[PlanEvaluation] = []

    for day in range(1, scenario.deadline + 1):
        weather = scenario.known_weather[day - 1]
        opponent_day = opponent_schedule[day - 1] if day <= len(opponent_schedule) else None
        next_states: dict[tuple[int, int, int], tuple[Fraction, tuple[DayDecision, ...]]] = {}
        for (location, used_water, used_food), (revenue, path) in states.items():
            options = [DayDecision(Action.STAY)]
            if weather is not Weather.SANDSTORM:
                options.extend(DayDecision(Action.WALK, node) for node in sorted(scenario.graph[location]))
            if location in scenario.mines:
                options.append(DayDecision(Action.MINE))

            for decision in options:
                destination = int(decision.destination) if decision.action is Action.WALK else location
                dw, df, daily_revenue = _daily_effect(
                    scenario, weather, location, decision, opponent_day
                )
                water = used_water + dw
                food = used_food + df
                if water * scenario.water.mass + food * scenario.food.mass > scenario.capacity:
                    continue
                if water * scenario.water.base_price + food * scenario.food.base_price > scenario.initial_cash:
                    continue
                decisions = path + (decision,)
                total_revenue = revenue + daily_revenue
                if destination == scenario.destination:
                    purchase_cost = (
                        water * scenario.water.base_price + food * scenario.food.base_price
                    )
                    terminals.append(
                        PlanEvaluation(
                            ActionPath(decisions),
                            water,
                            food,
                            day,
                            total_revenue,
                            Fraction(scenario.initial_cash - purchase_cost) + total_revenue,
                        )
                    )
                    continue
                remaining = scenario.deadline - day
                if shortest_distance(scenario, destination, scenario.destination) > remaining:
                    continue
                key = (destination, water, food)
                incumbent = next_states.get(key)
                if incumbent is None or total_revenue > incumbent[0]:
                    next_states[key] = (total_revenue, decisions)
        states = next_states

    if not terminals:
        raise RuntimeError("不存在可行响应")
    return max(
        terminals,
        key=lambda item: (
            item.final_value,
            -item.arrival_day,
            tuple((d.action.value, d.destination or 0) for d in item.path.decisions),
        ),
    )


def solve_level5_equilibrium(scenario: Scenario) -> Level5Equilibrium:
    """用最佳响应迭代寻找纯策略均衡，并显式复核单边偏离增益。"""
    player1 = best_response(scenario, None)
    seen: set[tuple[ActionPath, ActionPath]] = set()
    for _ in range(30):
        player2 = best_response(scenario, player1.path)
        player1_new = best_response(scenario, player2.path)
        profile = (player1_new.path, player2.path)
        if profile in seen:
            raise RuntimeError("最佳响应迭代发生循环，未找到纯策略均衡")
        seen.add(profile)
        player2_check = best_response(scenario, player1_new.path)
        if player1_new.path == player1.path and player2_check.path == player2.path:
            actual1 = evaluate_path(scenario, player1_new.path, player2.path)
            actual2 = evaluate_path(scenario, player2.path, player1_new.path)
            return Level5Equilibrium(
                actual1,
                actual2,
                best_response(scenario, player2.path).final_value - actual1.final_value,
                best_response(scenario, player1_new.path).final_value - actual2.final_value,
            )
        player1 = player1_new
    raise RuntimeError("最佳响应迭代未在 30 轮内收敛")
