from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean, median

from .model import Scenario
from .online import OnlinePolicy, simulate_policy
from .simulator import SimulationError
from .weather import IIDWeatherModel


@dataclass(frozen=True)
class EvaluationSummary:
    samples: int
    successes: int
    failures: int
    failure_rate: float
    mean_value: float | None
    median_value: float | None
    p05_value: float | None
    minimum_value: float | None
    maximum_value: float | None
    mean_arrival_day: float | None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(probability * (len(ordered) - 1))))
    return ordered[index]


def monte_carlo_evaluate(
    scenario: Scenario,
    policy: OnlinePolicy,
    weather_model: IIDWeatherModel,
    samples: int,
    seed: int,
) -> EvaluationSummary:
    rng = random.Random(seed)
    values: list[float] = []
    arrivals: list[int] = []
    failures = 0
    for _ in range(samples):
        weather = weather_model.sample(scenario.deadline, rng)
        try:
            run = simulate_policy(scenario, policy, weather)
        except (SimulationError, RuntimeError):
            failures += 1
            continue
        if not run.result.reached_destination or run.result.final_value is None:
            failures += 1
            continue
        values.append(run.result.final_value)
        arrivals.append(int(run.result.arrival_day))
    return EvaluationSummary(
        samples=samples,
        successes=len(values),
        failures=failures,
        failure_rate=failures / samples,
        mean_value=mean(values) if values else None,
        median_value=median(values) if values else None,
        p05_value=_quantile(values, 0.05) if values else None,
        minimum_value=min(values) if values else None,
        maximum_value=max(values) if values else None,
        mean_arrival_day=mean(arrivals) if arrivals else None,
    )

