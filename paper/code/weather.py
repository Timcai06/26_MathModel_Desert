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


@dataclass(frozen=True)
class MarkovWeatherModel:
    """有限状态一阶 Markov 天气模型。

    from_stationary 使用 P_rho = rho I + (1-rho) 1 p^T 构造转移矩阵，
    因此在改变连续天气聚集程度的同时严格保持长期边际分布 p 不变。
    """

    initial_probabilities: Mapping[Weather, float]
    transition_probabilities: Mapping[Weather, Mapping[Weather, float]]
    persistence: float | None = None

    def __post_init__(self) -> None:
        initial = {
            weather: float(self.initial_probabilities.get(weather, 0.0))
            for weather in Weather
        }
        if any(value < 0 for value in initial.values()):
            raise ValueError("初始天气概率不能为负")
        if abs(sum(initial.values()) - 1.0) > 1e-9:
            raise ValueError("初始天气概率之和必须为 1")

        transition: dict[Weather, dict[Weather, float]] = {}
        for origin in Weather:
            raw = self.transition_probabilities.get(origin)
            if raw is None:
                raise ValueError(f"缺少天气 {origin.value} 的转移概率")
            row = {destination: float(raw.get(destination, 0.0)) for destination in Weather}
            if any(value < 0 for value in row.values()):
                raise ValueError("天气转移概率不能为负")
            if abs(sum(row.values()) - 1.0) > 1e-9:
                raise ValueError(f"天气 {origin.value} 的转移概率之和必须为 1")
            transition[origin] = row

        object.__setattr__(self, "initial_probabilities", initial)
        object.__setattr__(self, "transition_probabilities", transition)

    @classmethod
    def from_stationary(
        cls,
        probabilities: Mapping[Weather, float],
        persistence: float,
    ) -> "MarkovWeatherModel":
        if not 0.0 <= persistence < 1.0:
            raise ValueError("持续性参数必须满足 0 <= rho < 1")
        stationary = {weather: float(probabilities.get(weather, 0.0)) for weather in Weather}
        if any(value < 0 for value in stationary.values()):
            raise ValueError("平稳天气概率不能为负")
        if abs(sum(stationary.values()) - 1.0) > 1e-9:
            raise ValueError("平稳天气概率之和必须为 1")
        transition = {
            origin: {
                destination: (
                    persistence * float(origin is destination)
                    + (1.0 - persistence) * stationary[destination]
                )
                for destination in Weather
            }
            for origin in Weather
        }
        return cls(stationary, transition, persistence)

    @property
    def support(self) -> tuple[Weather, ...]:
        return tuple(
            weather
            for weather in Weather
            if self.initial_probabilities[weather] > 0
        )

    def sample(self, days: int, rng: random.Random) -> tuple[Weather, ...]:
        if days < 0:
            raise ValueError("天气天数不能为负")
        if days == 0:
            return ()
        all_weather = tuple(Weather)
        first = rng.choices(
            all_weather,
            weights=[self.initial_probabilities[weather] for weather in all_weather],
            k=1,
        )[0]
        sequence = [first]
        for _ in range(1, days):
            previous = sequence[-1]
            sequence.append(
                rng.choices(
                    all_weather,
                    weights=[
                        self.transition_probabilities[previous][weather]
                        for weather in all_weather
                    ],
                    k=1,
                )[0]
            )
        return tuple(sequence)

    def sequence_probability(self, sequence: tuple[Weather, ...]) -> float:
        if not sequence:
            return 1.0
        probability = self.initial_probabilities[sequence[0]]
        for previous, current in zip(sequence, sequence[1:]):
            probability *= self.transition_probabilities[previous][current]
        return probability

    def enumerate(self, days: int) -> Iterable[tuple[tuple[Weather, ...], float]]:
        support = tuple(Weather)
        for sequence in product(support, repeat=days):
            probability = self.sequence_probability(tuple(sequence))
            if probability > 0:
                yield tuple(sequence), probability
