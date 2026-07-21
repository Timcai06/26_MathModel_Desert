from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from .model import Action, DayDecision, Plan, Scenario, Weather
from .simulator import SimulationResult, simulate


class OptimizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizationResult:
    plan: Plan
    simulation: SimulationResult
    solver_status: str
    doubled_objective: int


def _expr(terms: list[tuple[int, str]]) -> str:
    parts: list[str] = []
    for coefficient, variable in terms:
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        atom = variable if magnitude == 1 else f"{magnitude} {variable}"
        if not parts:
            parts.append(atom if coefficient > 0 else f"- {atom}")
        else:
            parts.append((" + " if coefficient > 0 else " - ") + atom)
    return "".join(parts) or "0"


def _build_lp(scenario: Scenario) -> str:
    if scenario.known_weather is None:
        raise OptimizationError("精确求解器只接受全程天气已知的关卡")
    days = scenario.deadline
    binaries: list[str] = []
    generals: list[str] = []
    bounds: list[str] = []
    constraints: list[str] = []
    action_by_day: dict[int, list[tuple[str, int, int, int, int, Action]]] = {}
    # tuple: variable, origin, destination, water consumption, food consumption, action

    for day in range(days + 1):
        for node in range(1, scenario.node_count + 1):
            binaries.append(f"x_{day}_{node}")

    for day, weather in enumerate(scenario.known_weather, start=1):
        actions: list[tuple[str, int, int, int, int, Action]] = []
        base_water = scenario.water.consumption[weather]
        base_food = scenario.food.consumption[weather]
        for node in range(1, scenario.node_count + 1):
            if node == scenario.destination:
                name = f"a_h_{day}"
                binaries.append(name)
                actions.append((name, node, node, 0, 0, Action.STAY))
                continue
            stay = f"a_s_{day}_{node}"
            binaries.append(stay)
            actions.append((stay, node, node, base_water, base_food, Action.STAY))
            if node in scenario.mines:
                mine = f"a_m_{day}_{node}"
                binaries.append(mine)
                actions.append((mine, node, node, 3 * base_water, 3 * base_food, Action.MINE))
            if weather is not Weather.SANDSTORM:
                for neighbour in sorted(scenario.graph[node]):
                    walk = f"a_w_{day}_{node}_{neighbour}"
                    binaries.append(walk)
                    actions.append((walk, node, neighbour, 2 * base_water, 2 * base_food, Action.WALK))
        action_by_day[day] = actions

    # 起始位置与截止日终点。
    for node in range(1, scenario.node_count + 1):
        constraints.append(f" start_{node}: x_0_{node} = {1 if node == scenario.start else 0}")
    constraints.append(f" finish: x_{days}_{scenario.destination} = 1")

    # 每日行动的流守恒；终点只有零消耗吸收自环，不能再次离开。
    for day in range(1, days + 1):
        actions = action_by_day[day]
        for node in range(1, scenario.node_count + 1):
            outgoing = [(1, name) for name, origin, _, _, _, _ in actions if origin == node]
            incoming = [(1, name) for name, _, destination, _, _, _ in actions if destination == node]
            constraints.append(
                f" out_{day}_{node}: {_expr(outgoing + [(-1, f'x_{day - 1}_{node}')])} = 0"
            )
            constraints.append(
                f" in_{day}_{node}: {_expr(incoming + [(-1, f'x_{day}_{node}')])} = 0"
            )

    # 物资、现金与采购。
    generals.extend(("bw_0", "bf_0", "rw_0", "rf_0"))
    bounds.extend((" 0 <= bw_0 <= 400", " 0 <= bf_0 <= 600"))
    constraints.append(" water_0: rw_0 - bw_0 = 0")
    constraints.append(" food_0: rf_0 - bf_0 = 0")
    constraints.append(
        f" cash_0_eq: cash_0 + {scenario.water.base_price} bw_0 + {scenario.food.base_price} bf_0 = {scenario.initial_cash}"
    )

    for day in range(days + 1):
        bounds.extend(
            (f" 0 <= rw_{day} <= 400", f" 0 <= rf_{day} <= 600", f" 0 <= cash_{day} <= {scenario.initial_cash + days * scenario.mine_income}")
        )
        constraints.append(
            f" load_{day}: {scenario.water.mass} rw_{day} + {scenario.food.mass} rf_{day} <= {scenario.capacity}"
        )

    for day in range(1, days + 1):
        generals.extend((f"rw_{day}", f"rf_{day}"))
        buy_water: list[str] = []
        buy_food: list[str] = []
        for village in sorted(scenario.villages):
            bw = f"bw_{day}_{village}"
            bf = f"bf_{day}_{village}"
            buy_water.append(bw)
            buy_food.append(bf)
            generals.extend((bw, bf))
            bounds.extend((f" 0 <= {bw} <= 400", f" 0 <= {bf} <= 600"))
            constraints.append(f" link_w_{day}_{village}: {bw} - 400 x_{day}_{village} <= 0")
            constraints.append(f" link_f_{day}_{village}: {bf} - 600 x_{day}_{village} <= 0")

        actions = action_by_day[day]
        water_terms = [(water, name) for name, _, _, water, _, _ in actions]
        food_terms = [(food, name) for name, _, _, _, food, _ in actions]
        water_balance = [(1, f"rw_{day}"), (-1, f"rw_{day - 1}")]
        water_balance.extend((-1, name) for name in buy_water)
        water_balance.extend(water_terms)
        food_balance = [(1, f"rf_{day}"), (-1, f"rf_{day - 1}")]
        food_balance.extend((-1, name) for name in buy_food)
        food_balance.extend(food_terms)
        constraints.append(f" water_{day}: {_expr(water_balance)} = 0")
        constraints.append(f" food_{day}: {_expr(food_balance)} = 0")

        # 采购在当日消耗后发生，因而采购前的结余也不得为负。
        constraints.append(
            f" prebuy_w_{day}: {_expr([(1, f'rw_{day}')] + [(-1, name) for name in buy_water])} >= 0"
        )
        constraints.append(
            f" prebuy_f_{day}: {_expr([(1, f'rf_{day}')] + [(-1, name) for name in buy_food])} >= 0"
        )
        # 未到终点时，日末水和食物至少各剩一箱。
        constraints.append(f" alive_w_{day}: rw_{day} + 400 x_{day}_{scenario.destination} >= 1")
        constraints.append(f" alive_f_{day}: rf_{day} + 600 x_{day}_{scenario.destination} >= 1")

        cash_terms = [(1, f"cash_{day}"), (-1, f"cash_{day - 1}")]
        cash_terms.extend((2 * scenario.water.base_price, name) for name in buy_water)
        cash_terms.extend((2 * scenario.food.base_price, name) for name in buy_food)
        cash_terms.extend(
            (-scenario.mine_income, name)
            for name, _, _, _, _, action in actions
            if action is Action.MINE
        )
        constraints.append(f" cash_{day}_eq: {_expr(cash_terms)} = 0")

    lines = [
        "Maximize",
        f" objective: 2 cash_{days} + 5 rw_{days} + 10 rf_{days}",
        "Subject To",
        *constraints,
        "Bounds",
        *bounds,
        "Binary",
        *[f" {name}" for name in binaries],
        "General",
        *[f" {name}" for name in generals],
        "End",
    ]
    return "\n".join(lines) + "\n"


def solve_known_weather(scenario: Scenario, time_limit: int = 120) -> OptimizationResult:
    """用 HiGHS 对时间扩展网络整数规划做全局精确求解。"""
    try:
        import highspy
    except ImportError as exc:
        raise OptimizationError("未安装 highspy；请使用 solution/.venv 中的 Python") from exc
    model = _build_lp(scenario)
    with tempfile.TemporaryDirectory(prefix=f"desert-level-{scenario.level}-") as tmp:
        model_path = Path(tmp) / "model.lp"
        model_path.write_text(model, encoding="utf-8")
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("time_limit", float(time_limit))
        if highs.readModel(str(model_path)) != highspy.HighsStatus.kOk:
            raise OptimizationError("HiGHS 无法读取生成的整数规划模型")
        highs.run()
        model_status = highs.getModelStatus()
        status = highs.modelStatusToString(model_status)
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise OptimizationError(f"HiGHS 未证明全局最优：{status}")
        lp = highs.getLp()
        solution = highs.getSolution()
        values = dict(zip(lp.col_names_, solution.col_value, strict=True))

    initial_water = round(values.get("bw_0", 0))
    initial_food = round(values.get("bf_0", 0))
    decisions: list[DayDecision] = []
    for day in range(1, scenario.deadline + 1):
        if values.get(f"a_h_{day}", 0) > 0.5:
            break
        buy_water = sum(round(values.get(f"bw_{day}_{village}", 0)) for village in scenario.villages)
        buy_food = sum(round(values.get(f"bf_{day}_{village}", 0)) for village in scenario.villages)
        selected: DayDecision | None = None
        for node in range(1, scenario.node_count + 1):
            if values.get(f"a_s_{day}_{node}", 0) > 0.5:
                selected = DayDecision(Action.STAY, buy_water=buy_water, buy_food=buy_food)
            if values.get(f"a_m_{day}_{node}", 0) > 0.5:
                selected = DayDecision(Action.MINE, buy_water=buy_water, buy_food=buy_food)
            for neighbour in scenario.graph[node]:
                if values.get(f"a_w_{day}_{node}_{neighbour}", 0) > 0.5:
                    selected = DayDecision(Action.WALK, neighbour, buy_water, buy_food)
        if selected is None:
            raise OptimizationError(f"无法从 CBC 解中恢复第 {day} 日行动")
        decisions.append(selected)
        if values.get(f"x_{day}_{scenario.destination}", 0) > 0.5:
            break

    plan = Plan(initial_water, initial_food, tuple(decisions))
    checked = simulate(scenario, plan)
    doubled = round(2 * checked.final_value) if checked.final_value is not None else -1
    return OptimizationResult(plan, checked, status, doubled)
