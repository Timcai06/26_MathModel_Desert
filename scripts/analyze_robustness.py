from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import mean

from desert.evaluation import evaluate_weather_sequences
from desert.level4_policies import MineThresholdPolicy
from desert.level4_search import candidate_grid
from desert.level6_oracle import RouteProfilePolicy, enumerate_shortest_routes
from desert.level6_policies import CoordinatedDirectPolicy
from desert.model import Weather
from desert.multiplayer import simulate_joint_policy
from desert.scenarios import get_scenario
from desert.weather import MarkovWeatherModel
from analyze_problem3 import MODELS, _wilson, evaluate_level6_policy


OUTPUT = Path("output/robustness")
SEED = 20261021
IN_SET_RHOS = (0.0, 0.15, 0.35)
STRESS_RHOS = IN_SET_RHOS + (0.70,)
RISK_BUDGETS = {"基准": 0.002, "不利": 0.005}


def _models(rhos: tuple[float, ...]) -> dict[tuple[str, float], MarkovWeatherModel]:
    return {
        (name, rho): MarkovWeatherModel.from_stationary(MODELS[name].probabilities, rho)
        for name in ("基准", "不利")
        for rho in rhos
    }


def _sample_suite(
    scenario_level: int,
    rhos: tuple[float, ...],
    samples: int,
    seed: int,
) -> dict[tuple[str, float], list[tuple[Weather, ...]]]:
    scenario = get_scenario(scenario_level)
    suite: dict[tuple[str, float], list[tuple[Weather, ...]]] = {}
    for index, (key, model) in enumerate(_models(rhos).items()):
        rng = random.Random(seed + index)
        suite[key] = [model.sample(scenario.deadline, rng) for _ in range(samples)]
    return suite


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weather_persistence_validation(samples: int) -> list[dict]:
    rows: list[dict] = []
    horizon = get_scenario(4).deadline
    for index, ((name, rho), model) in enumerate(_models(STRESS_RHOS).items()):
        rng = random.Random(SEED + 10 + index)
        sequences = [model.sample(horizon, rng) for _ in range(samples)]
        same = sum(
            first is second
            for sequence in sequences
            for first, second in zip(sequence, sequence[1:])
        )
        transitions = samples * (horizon - 1)
        storm_share = sum(
            weather is Weather.SANDSTORM for sequence in sequences for weather in sequence
        ) / (samples * horizon)
        longest_storm_runs: list[int] = []
        for sequence in sequences:
            longest = current = 0
            for weather in sequence:
                current = current + 1 if weather is Weather.SANDSTORM else 0
                longest = max(longest, current)
            longest_storm_runs.append(longest)
        theoretical_same = rho + (1 - rho) * sum(
            probability**2 for probability in MODELS[name].probabilities.values()
        )
        rows.append(
            {
                "weather_model": name,
                "persistence": rho,
                "samples": samples,
                "empirical_storm_share": storm_share,
                "empirical_same_weather_rate": same / transitions,
                "theoretical_same_weather_rate": theoretical_same,
                "mean_longest_storm_run": mean(longest_storm_runs),
                "probability_storm_run_at_least_3": sum(x >= 3 for x in longest_storm_runs)
                / samples,
                "in_ambiguity_set": rho in IN_SET_RHOS,
            }
        )
    return rows


def search_level4_robust(training_samples: int) -> tuple[MineThresholdPolicy, list[dict]]:
    scenario = get_scenario(4)
    suite = _sample_suite(4, IN_SET_RHOS, training_samples, SEED + 100)
    rows: list[dict] = []
    for candidate in candidate_grid(scenario):
        results = {
            key: evaluate_weather_sequences(scenario, candidate.policy(), sequences)
            for key, sequences in suite.items()
        }
        feasible = all(
            result.failure_rate <= RISK_BUDGETS[key[0]]
            for key, result in results.items()
        )
        rows.append(
            {
                **asdict(candidate),
                "training_samples_per_model": training_samples,
                "feasible": feasible,
                "worst_failure_rate": max(x.failure_rate for x in results.values()),
                "worst_mean_penalized_value": min(
                    x.mean_penalized_value for x in results.values()
                ),
                "worst_p05": min(x.penalized_p05_value for x in results.values()),
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    selected = max(
        feasible_rows,
        key=lambda row: (
            row["worst_mean_penalized_value"],
            row["worst_p05"],
            -row["initial_water"] - row["initial_food"],
        ),
    )
    policy = MineThresholdPolicy(
        int(selected["sandstorm_budget"]),
        int(selected["initial_water"]),
        int(selected["initial_food"]),
    )
    return policy, rows


def evaluate_level4_robustness(
    robust_policy: MineThresholdPolicy,
    samples: int,
) -> list[dict]:
    scenario = get_scenario(4)
    suite = _sample_suite(4, STRESS_RHOS, samples, SEED + 200)
    policies = {
        "IID效率策略_B5_240_240": MineThresholdPolicy(5, 240, 240),
        "分布鲁棒策略": robust_policy,
    }
    rows: list[dict] = []
    for (name, rho), sequences in suite.items():
        for policy_name, policy in policies.items():
            result = evaluate_weather_sequences(scenario, policy, sequences)
            water, food = policy.initial_purchase(scenario)
            rows.append(
                {
                    "weather_model": name,
                    "persistence": rho,
                    "in_ambiguity_set": rho in IN_SET_RHOS,
                    "policy": policy_name,
                    "initial_water": water,
                    "initial_food": food,
                    "sandstorm_budget": policy.sandstorm_budget,
                    **asdict(result),
                    "risk_budget": RISK_BUDGETS[name],
                    "wilson_pass": result.failure_wilson_high <= RISK_BUDGETS[name],
                }
            )
    return rows


def _direct_requirements(
    scenario_level: int,
    weather: tuple[Weather, ...],
    voluntary_waits: int,
) -> tuple[int, int]:
    scenario = get_scenario(scenario_level)
    distance = 8
    walks = water = food = 0
    waits_left = voluntary_waits
    for today in weather:
        if walks >= distance:
            break
        if today is Weather.SANDSTORM:
            factor = 1
        elif waits_left:
            factor = 1
            waits_left -= 1
        else:
            factor = 2
            walks += 1
        water += factor * scenario.water.consumption[today]
        food += factor * scenario.food.consumption[today]
    return water, food


def _requirement_value(water: int, food: int, required: tuple[int, int]) -> float:
    required_water, required_food = required
    if required_water > water or required_food > food:
        return 0.0
    return (
        10000
        - 2.5 * water
        - 5.0 * food
        - 2.5 * required_water
        - 5.0 * required_food
    )


def calibrate_direct_loads(
    samples: int,
) -> tuple[dict[int, tuple[int, int]], list[dict]]:
    suite = _sample_suite(6, IN_SET_RHOS, samples, SEED + 300)
    rows: list[dict] = []
    selected: dict[int, tuple[int, int]] = {}
    for wait in (0, 1, 2):
        requirements = {
            key: [_direct_requirements(6, sequence, wait) for sequence in sequences]
            for key, sequences in suite.items()
        }
        wait_rows: list[dict] = []
        for water in range(180, 241, 5):
            for food in range(180, 241, 5):
                if 3 * water + 2 * food > get_scenario(6).capacity:
                    continue
                feasible = True
                worst_mean = float("inf")
                worst_upper = 0.0
                for (name, _rho), values in requirements.items():
                    failures = sum(w > water or f > food for w, f in values)
                    upper = _wilson(failures, len(values))[1]
                    feasible &= upper <= RISK_BUDGETS[name]
                    worst_upper = max(worst_upper, upper)
                    worst_mean = min(
                        worst_mean,
                        mean(_requirement_value(water, food, value) for value in values),
                    )
                row = {
                    "voluntary_waits": wait,
                    "initial_water": water,
                    "initial_food": food,
                    "samples_per_model": samples,
                    "feasible": feasible,
                    "worst_wilson_high": worst_upper,
                    "worst_mean_value": worst_mean,
                }
                wait_rows.append(row)
                rows.append(row)
        feasible_rows = [row for row in wait_rows if row["feasible"]]
        if feasible_rows:
            best = max(feasible_rows, key=lambda row: row["worst_mean_value"])
            selected[wait] = (int(best["initial_water"]), int(best["initial_food"]))
    return selected, rows


def recommended_profile(loads_by_wait: dict[int, tuple[int, int]]) -> RouteProfilePolicy:
    direct = CoordinatedDirectPolicy()
    return RouteProfilePolicy(
        routes=direct.routes,
        initial_loads=(loads_by_wait[0], loads_by_wait[0], loads_by_wait[1]),
        voluntary_waits=(0, 0, 1),
    )


def evaluate_level6_robustness(
    robust_profile: RouteProfilePolicy,
    samples: int,
) -> list[dict]:
    suite = _sample_suite(6, STRESS_RHOS, samples, SEED + 400)
    policies = {
        "IID效率策略_185_185": CoordinatedDirectPolicy(185),
        "角色条件分布鲁棒策略": robust_profile,
    }
    rows: list[dict] = []
    for (name, rho), sequences in suite.items():
        for policy_name, policy in policies.items():
            result = evaluate_level6_policy(policy, sequences)
            rows.append(
                {
                    "weather_model": name,
                    "persistence": rho,
                    "in_ambiguity_set": rho in IN_SET_RHOS,
                    "policy": policy_name,
                    **result,
                    "risk_budget": RISK_BUDGETS[name],
                    "wilson_pass": result["max_role_failure_wilson_high"]
                    <= RISK_BUDGETS[name],
                }
            )
    return rows


def _evaluate_role(
    policy: RouteProfilePolicy,
    player: int,
    weather_sequences: list[tuple[Weather, ...]],
) -> dict:
    scenario = get_scenario(6)
    values: list[float] = []
    failures = 0
    for weather in weather_sequences:
        value = simulate_joint_policy(scenario, policy, weather).final_values[player]
        failures += value is None
        values.append(0.0 if value is None else float(value))
    low, high = _wilson(failures, len(values))
    return {
        "samples": len(values),
        "failures": failures,
        "failure_rate": failures / len(values),
        "failure_wilson_low": low,
        "failure_wilson_high": high,
        "mean_value": mean(values),
    }


def best_response_oracle(
    profile: RouteProfilePolicy,
    loads_by_wait: dict[int, tuple[int, int]],
    training_samples: int,
    test_samples: int,
) -> tuple[list[dict], list[dict]]:
    scenario = get_scenario(6)
    routes = enumerate_shortest_routes(scenario)
    worst_model = MarkovWeatherModel.from_stationary(MODELS["不利"].probabilities, 0.35)
    rng = random.Random(SEED + 500)
    training_weather = [
        worst_model.sample(scenario.deadline, rng) for _ in range(training_samples)
    ]
    screening: list[dict] = []
    selected: dict[int, RouteProfilePolicy] = {}
    for player in range(scenario.player_count):
        player_rows: list[tuple[dict, RouteProfilePolicy]] = []
        for route_index, route in enumerate(routes):
            for wait, (water, food) in sorted(loads_by_wait.items()):
                candidate = profile.unilateral_deviation(
                    player,
                    route=route,
                    initial_water=water,
                    initial_food=food,
                    voluntary_wait=wait,
                )
                result = _evaluate_role(candidate, player, training_weather)
                row = {
                    "player": player + 1,
                    "route_index": route_index,
                    "route": "-".join(map(str, route)),
                    "voluntary_waits": wait,
                    "initial_water": water,
                    "initial_food": food,
                    **result,
                    "training_risk_budget": RISK_BUDGETS["不利"],
                    "training_feasible": result["failure_rate"]
                    <= RISK_BUDGETS["不利"],
                }
                screening.append(row)
                player_rows.append((row, candidate))
        feasible = [item for item in player_rows if item[0]["training_feasible"]]
        selected[player] = max(feasible, key=lambda item: item[0]["mean_value"])[1]

    suite = _sample_suite(6, STRESS_RHOS, test_samples, SEED + 600)
    results: list[dict] = []
    for player in range(scenario.player_count):
        baseline = profile
        best_response = selected[player]
        for (name, rho), sequences in suite.items():
            base_result = _evaluate_role(baseline, player, sequences)
            best_result = _evaluate_role(best_response, player, sequences)
            results.append(
                {
                    "player": player + 1,
                    "weather_model": name,
                    "persistence": rho,
                    "in_ambiguity_set": rho in IN_SET_RHOS,
                    "baseline_mean_value": base_result["mean_value"],
                    "best_response_mean_value": best_result["mean_value"],
                    "exploitability": best_result["mean_value"]
                    - base_result["mean_value"],
                    "baseline_failure_rate": base_result["failure_rate"],
                    "best_response_failure_rate": best_result["failure_rate"],
                    "best_response_wilson_high": best_result["failure_wilson_high"],
                    "risk_budget": RISK_BUDGETS[name],
                    "best_response_risk_feasible": best_result["failure_wilson_high"]
                    <= RISK_BUDGETS[name],
                    "selected_route": "-".join(map(str, best_response.routes[player])),
                    "selected_waits": best_response.voluntary_waits[player],
                    "selected_water": best_response.initial_loads[player][0],
                    "selected_food": best_response.initial_loads[player][1],
                }
            )
    return screening, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather-samples", type=int, default=20000)
    parser.add_argument("--level4-training-samples", type=int, default=500)
    parser.add_argument("--load-calibration-samples", type=int, default=30000)
    parser.add_argument("--test-samples", type=int, default=30000)
    parser.add_argument("--oracle-training-samples", type=int, default=1500)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    weather_rows = weather_persistence_validation(args.weather_samples)
    level4_policy, level4_search = search_level4_robust(args.level4_training_samples)
    level4_rows = evaluate_level4_robustness(level4_policy, args.test_samples)
    loads_by_wait, load_rows = calibrate_direct_loads(args.load_calibration_samples)
    profile = recommended_profile(loads_by_wait)
    level6_rows = evaluate_level6_robustness(profile, args.test_samples)
    oracle_screening, exploitability = best_response_oracle(
        profile,
        loads_by_wait,
        args.oracle_training_samples,
        args.test_samples,
    )

    _write_csv(OUTPUT / "weather_persistence_validation.csv", weather_rows)
    _write_csv(OUTPUT / "level4_robust_search.csv", level4_search)
    _write_csv(OUTPUT / "level4_robust_comparison.csv", level4_rows)
    _write_csv(OUTPUT / "level6_role_load_calibration.csv", load_rows)
    _write_csv(OUTPUT / "level6_robust_comparison.csv", level6_rows)
    _write_csv(OUTPUT / "level6_oracle_screening.csv", oracle_screening)
    _write_csv(OUTPUT / "level6_exploitability.csv", exploitability)

    in_set_exploitability = [
        row["exploitability"] for row in exploitability if row["in_ambiguity_set"]
    ]
    payload = {
        "seed_base": SEED,
        "ambiguity_set": {
            "marginals": ["基准", "不利"],
            "persistence_values": list(IN_SET_RHOS),
            "out_of_set_stress_persistence": 0.70,
            "risk_budgets": RISK_BUDGETS,
        },
        "level4_selected": {
            "sandstorm_budget": level4_policy.sandstorm_budget,
            "initial_water": level4_policy.initial_water,
            "initial_food": level4_policy.initial_food,
        },
        "level6_selected": {
            "role_loads": profile.initial_loads,
            "voluntary_waits": profile.voluntary_waits,
            "routes": profile.routes,
        },
        "best_response_oracle": {
            "shortest_route_count": len(enumerate_shortest_routes(get_scenario(6))),
            "delay_choices": sorted(loads_by_wait),
            "candidate_count_per_player": len(enumerate_shortest_routes(get_scenario(6)))
            * len(loads_by_wait),
            "training_samples": args.oracle_training_samples,
            "test_samples_per_model": args.test_samples,
            "max_in_set_exploitability": max(in_set_exploitability),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT / "summary.json")


if __name__ == "__main__":
    main()
