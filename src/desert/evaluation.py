from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean, median, pstdev

from .model import Scenario
from .online import OnlinePolicy, simulate_policy
from .simulator import SimulationError
from .weather import IIDWeatherModel, MarkovWeatherModel


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
    mean_penalized_value: float
    penalized_p05_value: float
    failure_wilson_low: float
    failure_wilson_high: float
    mean_penalized_ci_low: float
    mean_penalized_ci_high: float


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(probability * (len(ordered) - 1))))
    return ordered[index]


def _wilson_interval(successes: int, samples: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """二项比例 Wilson 区间；这里的 ``successes`` 可表示失败次数。"""
    if samples <= 0:
        raise ValueError("样本数必须为正")
    proportion = successes / samples
    denominator = 1 + z * z / samples
    centre = (proportion + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / samples + z * z / (4 * samples * samples)
    ) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def evaluate_weather_sequences(
    scenario: Scenario,
    policy: OnlinePolicy,
    weather_sequences: list[tuple],
    failure_value: float = 0.0,
) -> EvaluationSummary:
    """在一组已配对天气序列上评估策略。

    成功条件均值用于解释到达后的财富水平；总体效用把失败记为
    ``failure_value``，用于选模，避免仅报告成功样本造成幸存者偏差。
    """
    values: list[float] = []
    arrivals: list[int] = []
    penalized_values: list[float] = []
    failures = 0
    for weather in weather_sequences:
        try:
            run = simulate_policy(scenario, policy, weather)
        except (SimulationError, RuntimeError):
            failures += 1
            penalized_values.append(failure_value)
            continue
        if not run.result.reached_destination or run.result.final_value is None:
            failures += 1
            penalized_values.append(failure_value)
            continue
        value = float(run.result.final_value)
        values.append(value)
        penalized_values.append(value)
        arrivals.append(int(run.result.arrival_day))

    samples = len(weather_sequences)
    if samples <= 0:
        raise ValueError("至少需要一条天气序列")
    failure_low, failure_high = _wilson_interval(failures, samples)
    penalized_mean = mean(penalized_values)
    standard_error = pstdev(penalized_values) / math.sqrt(samples)
    margin = 1.959963984540054 * standard_error
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
        mean_penalized_value=penalized_mean,
        penalized_p05_value=_quantile(penalized_values, 0.05),
        failure_wilson_low=failure_low,
        failure_wilson_high=failure_high,
        mean_penalized_ci_low=penalized_mean - margin,
        mean_penalized_ci_high=penalized_mean + margin,
    )


def monte_carlo_evaluate(
    scenario: Scenario,
    policy: OnlinePolicy,
    weather_model: IIDWeatherModel | MarkovWeatherModel,
    samples: int,
    seed: int,
) -> EvaluationSummary:
    rng = random.Random(seed)
    weather_sequences = [weather_model.sample(scenario.deadline, rng) for _ in range(samples)]
    return evaluate_weather_sequences(scenario, policy, weather_sequences)
