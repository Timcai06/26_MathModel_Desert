from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev

from desert.level5_game import ActionPath, PlanEvaluation, solve_level5_equilibrium
from desert.level6_policies import (
    CongestionAwareEquilibriumPolicy,
    CoordinatedDirectPolicy,
    CoordinatedMinePolicy,
    MirrorMinePolicy,
)
from desert.model import Action, Weather
from desert.multiplayer import JointOnlinePolicy, simulate_joint_policy
from desert.scenarios import get_scenario
from desert.weather import IIDWeatherModel


OUTPUT = Path("output/problem3")
SEED = 20260921


MODELS = {
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


def _fraction(value: Fraction) -> int | float:
    return int(value) if value.denominator == 1 else float(value)


def _path_payload(plan: PlanEvaluation) -> dict:
    return {
        "initial_water": plan.initial_water,
        "initial_food": plan.initial_food,
        "arrival_day": plan.arrival_day,
        "mining_revenue": _fraction(plan.mining_revenue),
        "final_value": _fraction(plan.final_value),
        "actions": [
            {"action": decision.action.value, "destination": decision.destination}
            for decision in plan.path.decisions
        ],
    }


def analyze_level5() -> dict:
    scenario = get_scenario(5)
    equilibrium = solve_level5_equilibrium(scenario)
    return {
        "method": "complete finite-horizon best-response dynamic programming",
        "equilibrium_type": "asymmetric pure-strategy Nash equilibrium",
        "player1": _path_payload(equilibrium.player1),
        "player2": _path_payload(equilibrium.player2),
        "player1_max_unilateral_gain": _fraction(equilibrium.player1_deviation_gain),
        "player2_max_unilateral_gain": _fraction(equilibrium.player2_deviation_gain),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def _wilson(events: int, samples: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = events / samples
    denominator = 1 + z * z / samples
    centre = (p + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def evaluate_level6_policy(
    policy: JointOnlinePolicy,
    weather_sequences: list[tuple[Weather, ...]],
) -> dict:
    scenario = get_scenario(6)
    individual_values: list[float] = []
    role_values: list[list[float]] = [[] for _ in range(scenario.player_count)]
    role_failures = [0 for _ in range(scenario.player_count)]
    arrivals: list[int] = []
    failed_players = 0
    failed_sessions = 0
    shared_edge_player_days = 0
    shared_mine_player_days = 0
    solo_mine_player_days = 0

    for weather in weather_sequences:
        result = simulate_joint_policy(scenario, policy, weather)
        session_failed = False
        for role, value in enumerate(result.final_values):
            numeric = 0.0 if value is None else float(value)
            individual_values.append(numeric)
            role_values[role].append(numeric)
            if value is None:
                failed_players += 1
                role_failures[role] += 1
                session_failed = True
            else:
                arrival = result.records[-1].states[role].arrival_day
                assert arrival is not None
                arrivals.append(arrival)
        failed_sessions += int(session_failed)

        for previous, record in zip(result.records, result.records[1:]):
            edge_groups: dict[tuple[int, int], int] = {}
            mine_groups: dict[int, int] = {}
            for state, decision in zip(previous.states, record.decisions):
                if decision is None:
                    continue
                if decision.action is Action.WALK:
                    key = (state.location, int(decision.destination))
                    edge_groups[key] = edge_groups.get(key, 0) + 1
                elif decision.action is Action.MINE:
                    mine_groups[state.location] = mine_groups.get(state.location, 0) + 1
            shared_edge_player_days += sum(size for size in edge_groups.values() if size >= 2)
            shared_mine_player_days += sum(size for size in mine_groups.values() if size >= 2)
            solo_mine_player_days += sum(1 for size in mine_groups.values() if size == 1)

    player_observations = len(individual_values)
    player_low, player_high = _wilson(failed_players, player_observations)
    session_low, session_high = _wilson(failed_sessions, len(weather_sequences))
    value_mean = mean(individual_values)
    margin = 1.959963984540054 * pstdev(individual_values) / math.sqrt(player_observations)
    role_means = [mean(values) for values in role_values]
    role_intervals = [_wilson(events, len(weather_sequences)) for events in role_failures]
    return {
        "samples": len(weather_sequences),
        "player_observations": player_observations,
        "player_failure_rate": failed_players / player_observations,
        "player_failure_wilson_low": player_low,
        "player_failure_wilson_high": player_high,
        "session_failure_rate": failed_sessions / len(weather_sequences),
        "session_failure_wilson_low": session_low,
        "session_failure_wilson_high": session_high,
        "mean_individual_value": value_mean,
        "mean_individual_ci_low": value_mean - margin,
        "mean_individual_ci_high": value_mean + margin,
        "p05_individual_value": _quantile(individual_values, 0.05),
        "role1_mean": role_means[0],
        "role2_mean": role_means[1],
        "role3_mean": role_means[2],
        "role_mean_gap": max(role_means) - min(role_means),
        "role1_failure_rate": role_failures[0] / len(weather_sequences),
        "role2_failure_rate": role_failures[1] / len(weather_sequences),
        "role3_failure_rate": role_failures[2] / len(weather_sequences),
        "role1_failure_wilson_high": role_intervals[0][1],
        "role2_failure_wilson_high": role_intervals[1][1],
        "role3_failure_wilson_high": role_intervals[2][1],
        "max_role_failure_wilson_high": max(high for _low, high in role_intervals),
        "mean_arrival_day": mean(arrivals) if arrivals else None,
        "shared_edge_player_days_per_session": shared_edge_player_days / len(weather_sequences),
        "shared_mine_player_days_per_session": shared_mine_player_days / len(weather_sequences),
        "solo_mine_player_days_per_session": solo_mine_player_days / len(weather_sequences),
    }


def search_direct_load(training_samples: int = 3000) -> tuple[int, list[dict]]:
    scenario = get_scenario(6)
    models = {
        "基准": MODELS["基准"],
        "不利": MODELS["不利"],
    }
    paired: dict[str, list[tuple[Weather, ...]]] = {}
    for index, (name, model) in enumerate(models.items()):
        rng = random.Random(SEED - 100 + index)
        paired[name] = [model.sample(scenario.deadline, rng) for _ in range(training_samples)]
    rows: list[dict] = []
    for boxes in range(180, 221, 5):
        summaries = {
            name: evaluate_level6_policy(CoordinatedDirectPolicy(boxes), weather)
            for name, weather in paired.items()
        }
        rows.append(
            {
                "initial_water": boxes,
                "initial_food": boxes,
                "training_samples": training_samples,
                "baseline_failure_rate": summaries["基准"]["player_failure_rate"],
                "baseline_mean_value": summaries["基准"]["mean_individual_value"],
                "adverse_failure_rate": summaries["不利"]["player_failure_rate"],
                "adverse_mean_value": summaries["不利"]["mean_individual_value"],
                "adverse_p05": summaries["不利"]["p05_individual_value"],
            }
        )
    feasible = [
        row
        for row in rows
        if row["baseline_failure_rate"] <= 0.002
        and row["adverse_failure_rate"] <= 0.005
    ]
    selected = max(feasible, key=lambda row: row["adverse_mean_value"])
    return int(selected["initial_water"]), rows


def analyze_level6(samples: int, direct_boxes: int) -> list[dict]:
    policies: dict[str, JointOnlinePolicy] = {
        "镜像挖矿": MirrorMinePolicy(),
        "分路直达": CoordinatedDirectPolicy(direct_boxes),
        "拥挤感知非合作": CongestionAwareEquilibriumPolicy(),
        "合作轮换": CoordinatedMinePolicy(),
    }
    rows: list[dict] = []
    for model_index, (model_name, model) in enumerate(MODELS.items()):
        rng = random.Random(SEED + model_index)
        weather = [model.sample(get_scenario(6).deadline, rng) for _ in range(samples)]
        for policy_name, policy in policies.items():
            rows.append(
                {
                    "weather_model": model_name,
                    "policy": policy_name,
                    **evaluate_level6_policy(policy, weather),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    level5 = analyze_level5()
    direct_boxes, load_search = search_direct_load()
    level6 = analyze_level6(args.samples, direct_boxes)
    _write_csv(OUTPUT / "level6_load_search.csv", load_search)
    _write_csv(OUTPUT / "level6_oos_comparison.csv", level6)
    payload = {
        "seed_base": SEED,
        "failure_terminal_value": 0,
        "level5": level5,
        "level6_load_selection": {
            "candidate_count": len(load_search),
            "training_samples_per_distribution": load_search[0]["training_samples"],
            "risk_budget": {
                "baseline_player_failure_rate": 0.002,
                "adverse_player_failure_rate": 0.005,
            },
            "selected_initial_water": direct_boxes,
            "selected_initial_food": direct_boxes,
        },
        "level6": level6,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT / "summary.json")


if __name__ == "__main__":
    main()
