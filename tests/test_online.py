import random
import unittest

from desert.level4_policies import DirectRobustPolicy, shortest_distance
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


if __name__ == "__main__":
    unittest.main()
