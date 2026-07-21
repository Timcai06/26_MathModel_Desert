from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable

from .evaluation import EvaluationSummary, evaluate_weather_sequences
from .level4_policies import MineThresholdPolicy
from .model import Scenario, Weather
from .weather import IIDWeatherModel


@dataclass(frozen=True)
class Level4Candidate:
    sandstorm_budget: int
    initial_water: int
    initial_food: int

    def policy(self) -> MineThresholdPolicy:
        return MineThresholdPolicy(
            self.sandstorm_budget,
            initial_water=self.initial_water,
            initial_food=self.initial_food,
        )


@dataclass(frozen=True)
class CandidateScore:
    candidate: Level4Candidate
    baseline: EvaluationSummary
    adverse: EvaluationSummary
    robust_score: float

    def as_row(self) -> dict:
        return {
            **asdict(self.candidate),
            "robust_score": self.robust_score,
            "baseline_failure_rate": self.baseline.failure_rate,
            "baseline_mean_penalized_value": self.baseline.mean_penalized_value,
            "baseline_p05": self.baseline.penalized_p05_value,
            "adverse_failure_rate": self.adverse.failure_rate,
            "adverse_mean_penalized_value": self.adverse.mean_penalized_value,
            "adverse_p05": self.adverse.penalized_p05_value,
        }


def candidate_grid(scenario: Scenario) -> Iterable[Level4Candidate]:
    """覆盖现有 240/240 策略附近的可解释整数网格。"""
    for budget in range(3, 10):
        for water in range(180, 281, 10):
            for food in range(180, 281, 10):
                if water * scenario.water.mass + food * scenario.food.mass > scenario.capacity:
                    continue
                purchase_cost = water * scenario.water.base_price + food * scenario.food.base_price
                if purchase_cost > scenario.initial_cash:
                    continue
                yield Level4Candidate(budget, water, food)


def _paired_samples(
    model: IIDWeatherModel,
    horizon: int,
    samples: int,
    seed: int,
) -> list[tuple[Weather, ...]]:
    rng = random.Random(seed)
    return [model.sample(horizon, rng) for _ in range(samples)]


def search_level4(
    scenario: Scenario,
    training_samples: int = 2000,
    seed: int = 20260721,
) -> list[CandidateScore]:
    baseline_model = IIDWeatherModel(
        {Weather.SUNNY: 0.45, Weather.HOT: 0.45, Weather.SANDSTORM: 0.10}
    )
    adverse_model = IIDWeatherModel(
        {Weather.SUNNY: 0.35, Weather.HOT: 0.50, Weather.SANDSTORM: 0.15}
    )
    baseline_weather = _paired_samples(baseline_model, scenario.deadline, training_samples, seed)
    adverse_weather = _paired_samples(adverse_model, scenario.deadline, training_samples, seed + 1)

    scores: list[CandidateScore] = []
    for candidate in candidate_grid(scenario):
        policy = candidate.policy()
        baseline = evaluate_weather_sequences(scenario, policy, baseline_weather)
        adverse = evaluate_weather_sequences(scenario, policy, adverse_weather)
        # 以两种分布中的较低总体效用为主，另对失败率作显式惩罚。
        robust_score = min(
            baseline.mean_penalized_value,
            adverse.mean_penalized_value,
        ) - 20000 * max(baseline.failure_rate, adverse.failure_rate)
        scores.append(CandidateScore(candidate, baseline, adverse, robust_score))
    return sorted(
        scores,
        key=lambda item: (
            item.robust_score,
            item.adverse.penalized_p05_value,
            -item.candidate.initial_water - item.candidate.initial_food,
        ),
        reverse=True,
    )
