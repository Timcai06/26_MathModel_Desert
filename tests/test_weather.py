from __future__ import annotations

import random
import unittest

from desert.model import Weather
from desert.weather import MarkovWeatherModel


BASELINE = {
    Weather.SUNNY: 0.45,
    Weather.HOT: 0.45,
    Weather.SANDSTORM: 0.10,
}


class MarkovWeatherTests(unittest.TestCase):
    def test_stationary_mixture_preserves_distribution(self) -> None:
        model = MarkovWeatherModel.from_stationary(BASELINE, 0.7)
        for destination in Weather:
            propagated = sum(
                BASELINE[origin] * model.transition_probabilities[origin][destination]
                for origin in Weather
            )
            self.assertAlmostEqual(propagated, BASELINE[destination])

    def test_persistence_increases_same_weather_probability(self) -> None:
        iid = MarkovWeatherModel.from_stationary(BASELINE, 0.0)
        persistent = MarkovWeatherModel.from_stationary(BASELINE, 0.7)
        for weather in Weather:
            self.assertGreater(
                persistent.transition_probabilities[weather][weather],
                iid.transition_probabilities[weather][weather],
            )

    def test_sampling_is_reproducible(self) -> None:
        model = MarkovWeatherModel.from_stationary(BASELINE, 0.35)
        self.assertEqual(
            model.sample(30, random.Random(20261001)),
            model.sample(30, random.Random(20261001)),
        )

    def test_sequence_probability_uses_transitions(self) -> None:
        model = MarkovWeatherModel.from_stationary(BASELINE, 0.5)
        sequence = (Weather.SANDSTORM, Weather.SANDSTORM, Weather.HOT)
        expected = (
            BASELINE[Weather.SANDSTORM]
            * model.transition_probabilities[Weather.SANDSTORM][Weather.SANDSTORM]
            * model.transition_probabilities[Weather.SANDSTORM][Weather.HOT]
        )
        self.assertAlmostEqual(model.sequence_probability(sequence), expected)


if __name__ == "__main__":
    unittest.main()
