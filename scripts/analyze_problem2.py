from __future__ import annotations

import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path

from desert.evaluation import monte_carlo_evaluate
from desert.level3_mdp import Level3MDPSolver
from desert.level4_policies import DirectRobustPolicy, MineThresholdPolicy
from desert.model import Weather
from desert.online import simulate_policy
from desert.scenarios import get_scenario
from desert.weather import IIDWeatherModel


OUTPUT = Path("output/problem2")
SEED = 20260721


def _weighted_quantile(values: list[tuple[float, float]], probability: float) -> float:
    cumulative = 0.0
    for value, weight in sorted(values):
        cumulative += weight
        if cumulative + 1e-12 >= probability:
            return value
    return max(value for value, _ in values)


def analyze_level3() -> list[dict]:
    scenario = get_scenario(3)
    rows: list[dict] = []
    for hot_probability in (0.2, 0.5, 0.8):
        model = IIDWeatherModel(
            {Weather.SUNNY: 1 - hot_probability, Weather.HOT: hot_probability}
        )
        solver = Level3MDPSolver(scenario, model)
        solution = solver.solve()
        weighted_values: list[tuple[float, float]] = []
        failures = 0
        expected_check = 0.0
        for weather, probability in model.enumerate(scenario.deadline):
            run = simulate_policy(scenario, solution.policy, weather)
            if not run.result.reached_destination or run.result.final_value is None:
                failures += 1
                continue
            value = float(run.result.final_value)
            weighted_values.append((value, probability))
            expected_check += probability * value
        sunny_run = simulate_policy(scenario, solution.policy, (Weather.SUNNY,) * scenario.deadline)
        hot_run = simulate_policy(scenario, solution.policy, (Weather.HOT,) * scenario.deadline)
        row = {
            "hot_probability": hot_probability,
            "initial_water": solution.initial_water,
            "initial_food": solution.initial_food,
            "expected_final_value": solution.expected_final_value,
            "enumerated_expected_value": expected_check,
            "p05_value": _weighted_quantile(weighted_values, 0.05),
            "minimum_value": min(value for value, _ in weighted_values),
            "maximum_value": max(value for value, _ in weighted_values),
            "failed_weather_sequences": failures,
            "states_evaluated": solution.states_evaluated,
            "all_sunny_arrival_day": sunny_run.result.arrival_day,
            "all_sunny_final_value": sunny_run.result.final_value,
            "all_sunny_actions": "-".join(d.action.value for d in sunny_run.plan.decisions),
            "all_hot_arrival_day": hot_run.result.arrival_day,
            "all_hot_final_value": hot_run.result.final_value,
            "all_hot_actions": "-".join(d.action.value for d in hot_run.plan.decisions),
        }
        if abs(solution.expected_final_value - expected_check) > 1e-6:
            raise RuntimeError("第三关 Bellman 值与全枚举回放不一致")
        rows.append(row)
        del solution, solver
        gc.collect()
    return rows


def analyze_level4(samples: int = 5000) -> list[dict]:
    scenario = get_scenario(4)
    models = {
        "低沙暴": IIDWeatherModel(
            {Weather.SUNNY: 0.50, Weather.HOT: 0.45, Weather.SANDSTORM: 0.05}
        ),
        "基准": IIDWeatherModel(
            {Weather.SUNNY: 0.45, Weather.HOT: 0.45, Weather.SANDSTORM: 0.10}
        ),
        "不利": IIDWeatherModel(
            {Weather.SUNNY: 0.35, Weather.HOT: 0.50, Weather.SANDSTORM: 0.15}
        ),
    }
    policies = {
        "直达_B4": DirectRobustPolicy(4),
        "挖矿_B4": MineThresholdPolicy(4),
        "挖矿_B6": MineThresholdPolicy(6),
    }
    rows: list[dict] = []
    for model_name, model in models.items():
        for policy_name, policy in policies.items():
            summary = monte_carlo_evaluate(scenario, policy, model, samples, SEED)
            initial_water, initial_food = policy.initial_purchase(scenario)
            rows.append(
                {
                    "weather_model": model_name,
                    "policy": policy_name,
                    "sunny_probability": model.probabilities[Weather.SUNNY],
                    "hot_probability": model.probabilities[Weather.HOT],
                    "sandstorm_probability": model.probabilities[Weather.SANDSTORM],
                    "initial_water": initial_water,
                    "initial_food": initial_food,
                    **asdict(summary),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    level3 = analyze_level3()
    level4 = analyze_level4()
    _write_csv(OUTPUT / "level3_mdp_sensitivity.csv", level3)
    _write_csv(OUTPUT / "level4_policy_comparison.csv", level4)
    summary = {
        "method": {
            "level3": "finite-horizon exact MDP; current weather observed; robust feasibility on weather support",
            "level4": "rule-based robust threshold policies with paired out-of-sample Monte Carlo",
        },
        "seed": SEED,
        "level3": level3,
        "level4": level4,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT / "summary.json")


if __name__ == "__main__":
    main()
