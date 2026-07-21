import random
import unittest

from desert.evaluation import evaluate_weather_sequences
from desert.level4_policies import DirectRobustPolicy, MineThresholdPolicy, shortest_distance
from desert.level4_mdp import Level4FixedLoadMDP
from desert.level4_search import candidate_grid
from desert.model import Weather
from desert.online import simulate_policy
from desert.scenarios import get_scenario
from desert.weather import IIDWeatherModel


class OnlineTests(unittest.TestCase):
    def test_iid_weather_is_reproducible(self) -> None:
        model = IIDWeatherModel({Weather.SUNNY: 0.5, Weather.HOT: 0.5})
        self.assertEqual(model.sample(10, random.Random(7)), model.sample(10, random.Random(7)))

    def test_level3_direct_policy_survives_all_hot(self) -> None:
        scenario = get_scenario(3)
        run = simulate_policy(
            scenario,
            DirectRobustPolicy(0),
            (Weather.HOT,) * scenario.deadline,
        )
        self.assertEqual(run.result.arrival_day, 3)
        self.assertEqual(run.result.final_value, 9190)

    def test_level4_distances(self) -> None:
        scenario = get_scenario(4)
        self.assertEqual(shortest_distance(scenario, 1, 25), 8)
        self.assertEqual(shortest_distance(scenario, 1, 18), 5)
        self.assertEqual(shortest_distance(scenario, 18, 25), 3)

    def test_level4_candidate_grid_respects_capacity(self) -> None:
        scenario = get_scenario(4)
        candidates = list(candidate_grid(scenario))
        self.assertGreater(len(candidates), 100)
        for candidate in candidates:
            self.assertLessEqual(3 * candidate.initial_water + 2 * candidate.initial_food, 1200)

    def test_failure_is_included_in_penalized_value(self) -> None:
        scenario = get_scenario(4)
        summary = evaluate_weather_sequences(
            scenario,
            DirectRobustPolicy(0),
            [(Weather.SANDSTORM,) * scenario.deadline],
        )
        self.assertEqual(summary.failure_rate, 1)
        self.assertEqual(summary.mean_penalized_value, 0)

    def test_level4_exact_benchmark_matches_all_sunny_threshold_replay(self) -> None:
        scenario = get_scenario(4)
        model = IIDWeatherModel({Weather.SUNNY: 1.0})
        result = Level4FixedLoadMDP(scenario, model, 240, 240).solve(5)
        replay = simulate_policy(scenario, MineThresholdPolicy(5, 240, 240), (Weather.SUNNY,) * scenario.deadline)
        self.assertGreaterEqual(result.optimal.expected_final_value, result.threshold.expected_final_value)
        self.assertEqual(result.threshold.failure_probability, 0)
        self.assertEqual(result.threshold.expected_final_value, replay.result.final_value)


if __name__ == "__main__":
    unittest.main()
