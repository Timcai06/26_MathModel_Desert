from __future__ import annotations

import argparse
import json
from pathlib import Path

from desert.optimizer import solve_known_weather
from desert.scenarios import get_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="精确求解全程天气已知的穿越沙漠关卡")
    parser.add_argument("levels", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = []
    for level in args.levels:
        result = solve_known_weather(get_scenario(level), args.time_limit)
        item = {
            "level": level,
            "solver_status": result.solver_status,
            "final_value": result.simulation.final_value,
            "arrival_day": result.simulation.arrival_day,
            "initial_water": result.plan.initial_water,
            "initial_food": result.plan.initial_food,
            "records": [
                {
                    "day": record.day,
                    "location": record.location,
                    "cash": record.cash,
                    "water": record.water,
                    "food": record.food,
                    "weather": record.weather.value if record.weather else None,
                    "action": record.action.value if record.action else None,
                    "buy_water": record.buy_water,
                    "buy_food": record.buy_food,
                }
                for record in result.simulation.records
            ],
        }
        payload.append(item)
        print(
            f"第{level}关：{result.solver_status}；第{item['arrival_day']}天到达；"
            f"初始水/食物={item['initial_water']}/{item['initial_food']}；最终价值={item['final_value']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
