from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from desert.evaluation import monte_carlo_evaluate
from desert.level4_policies import DirectRobustPolicy, MineThresholdPolicy
from desert.level4_search import CandidateScore, search_level4
from desert.model import Weather
from desert.scenarios import get_scenario
from desert.weather import IIDWeatherModel


OUTPUT = Path("output/problem2")
TRAINING_SEED = 20260721
TEST_SEED = 20260821


WEATHER_MODELS = {
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select(scores: list[CandidateScore]) -> CandidateScore:
    """先满足训练集风险预算，再最大化不利分布下的总体效用。"""
    feasible = [
        item
        for item in scores
        if item.baseline.failure_rate <= 0.002 and item.adverse.failure_rate <= 0.005
    ]
    pool = feasible or scores
    return max(
        pool,
        key=lambda item: (
            item.adverse.mean_penalized_value,
            item.baseline.mean_penalized_value,
            item.adverse.penalized_p05_value,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=20000)
    args = parser.parse_args()

    scenario = get_scenario(4)
    scores = search_level4(scenario, args.training_samples, TRAINING_SEED)
    selected = _select(scores)
    selected_policy = selected.candidate.policy()

    policies = {
        "直达_B4": DirectRobustPolicy(4),
        "原挖矿_B4": MineThresholdPolicy(4),
        "原挖矿_B6": MineThresholdPolicy(6),
        "网格搜索入选": selected_policy,
    }
    oos_rows: list[dict] = []
    for model_index, (model_name, model) in enumerate(WEATHER_MODELS.items()):
        for policy_name, policy in policies.items():
            summary = monte_carlo_evaluate(
                scenario,
                policy,
                model,
                args.test_samples,
                TEST_SEED + model_index,
            )
            initial_water, initial_food = policy.initial_purchase(scenario)
            oos_rows.append(
                {
                    "weather_model": model_name,
                    "policy": policy_name,
                    "initial_water": initial_water,
                    "initial_food": initial_food,
                    "sandstorm_budget": policy.sandstorm_budget,
                    **asdict(summary),
                }
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    leaderboard = [item.as_row() for item in scores]
    _write_csv(OUTPUT / "level4_parameter_search.csv", leaderboard)
    _write_csv(OUTPUT / "level4_oos_validation.csv", oos_rows)
    payload = {
        "selection_protocol": {
            "candidate_count": len(scores),
            "training_samples_per_distribution": args.training_samples,
            "training_seed": TRAINING_SEED,
            "risk_budget": {
                "baseline_failure_rate": 0.002,
                "adverse_failure_rate": 0.005,
            },
            "test_samples_per_policy_distribution": args.test_samples,
            "test_seed_base": TEST_SEED,
            "failure_terminal_value": 0,
        },
        "selected_training_result": selected.as_row(),
        "top10_training": leaderboard[:10],
        "out_of_sample": oos_rows,
    }
    (OUTPUT / "level4_optimization.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT / "level4_optimization.json")


if __name__ == "__main__":
    main()
