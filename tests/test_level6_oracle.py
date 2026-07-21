import unittest

from desert.level6_oracle import RouteProfilePolicy, enumerate_shortest_routes
from desert.level6_policies import CoordinatedDirectPolicy
from desert.model import Weather
from desert.multiplayer import simulate_joint_policy
from desert.scenarios import get_scenario


class Level6OracleTests(unittest.TestCase):
    def test_square_grid_has_seventy_shortest_routes(self) -> None:
        scenario = get_scenario(6)
        routes = enumerate_shortest_routes(scenario)
        self.assertEqual(len(routes), 70)
        self.assertTrue(all(route[0] == 1 and route[-1] == 25 for route in routes))
        self.assertTrue(all(len(route) == 9 for route in routes))

    def test_route_profile_reproduces_direct_policy(self) -> None:
        scenario = get_scenario(6)
        direct = CoordinatedDirectPolicy(185)
        profile = RouteProfilePolicy(
            routes=direct.routes,
            initial_loads=((185, 185),) * 3,
            voluntary_waits=(0, 0, 1),
        )
        weather = (Weather.SUNNY,) * scenario.deadline
        expected = simulate_joint_policy(scenario, direct, weather)
        actual = simulate_joint_policy(scenario, profile, weather)
        self.assertEqual(actual.final_values, expected.final_values)
        self.assertEqual(actual.records, expected.records)

    def test_unilateral_deviation_changes_only_selected_role(self) -> None:
        scenario = get_scenario(6)
        routes = enumerate_shortest_routes(scenario)
        profile = RouteProfilePolicy(
            routes=(routes[0], routes[-1], routes[0]),
            initial_loads=((225, 225),) * 3,
            voluntary_waits=(0, 0, 1),
        )
        changed = profile.unilateral_deviation(
            1, route=routes[1], initial_water=230, voluntary_wait=2
        )
        self.assertEqual(changed.routes[0], profile.routes[0])
        self.assertEqual(changed.routes[2], profile.routes[2])
        self.assertEqual(changed.initial_loads[1], (230, 225))
        self.assertEqual(changed.voluntary_waits, (0, 2, 1))


if __name__ == "__main__":
    unittest.main()
