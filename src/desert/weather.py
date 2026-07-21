from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random
from typing import Iterable, Mapping

from .model import Weather


@dataclass(frozen=True)
class IIDWeatherModel:
    probabilities: Mapping[Weather, float]

    def __post_init__(self) -> None:
        probabilities = {weather: float(self.probabilities.get(weather, 0.0)) for weather in Weather}
        if any(value < 0 for value in probabilities.values()):
            raise ValueError("天气概率不能为负")
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("天气概率之和必须为 1")
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def support(self) -> tuple[Weather, ...]:
        return tuple(weather for weather in Weather if self.probabilities[weather] > 0)

    def sample(self, days: int, rng: random.Random) -> tuple[Weather, ...]:
        support = self.support
        weights = [self.probabilities[weather] for weather in support]
        return tuple(rng.choices(support, weights=weights, k=days))

    def enumerate(self, days: int) -> Iterable[tuple[tuple[Weather, ...], float]]:
        support = self.support
        for sequence in product(support, repeat=days):
            probability = 1.0
            for weather in sequence:
                probability *= self.probabilities[weather]
            yield tuple(sequence), probability

