import unittest

from desert.level6_policies import (
    CongestionAwareEquilibriumPolicy,
    CoordinatedDirectPolicy,
    CoordinatedMinePolicy,
    MirrorMinePolicy,
)
from desert.model import Weather
from desert.multiplayer import simulate_joint_policy
from desert.scenarios import get_scenario


class MultiplayerTests(unittest.TestCase):
    def test_coordinated_direct_profile_arrives_in_all_sunny_weather(self) -> None:
        scenario = get_scenario(6)
        result = simulate_joint_policy(
            scenario, CoordinatedDirectPolicy(), (Weather.SUNNY,) * scenario.deadline
        )
        self.assertTrue(all(value is not None for value in result.final_values))

    def test_mirror_profile_pays_edge_congestion(self) -> None:
        scenario = get_scenario(6)
        result = simulate_joint_policy(
            scenario, MirrorMinePolicy(), (Weather.SUNNY,) * scenario.deadline
        )
        first_day = result.records[1].states
        self.assertEqual(first_day[0].water, 222)
        self.assertEqual(first_day[0].food, 216)

    def test_coordinated_mining_never_assigns_two_miners_same_day(self) -> None:
        scenario = get_scenario(6)
        result = simulate_joint_policy(
            scenario, CoordinatedMinePolicy(), (Weather.SUNNY,) * scenario.deadline
        )
        for record in result.records:
            mines = sum(
                decision is not None and decision.action.value == "挖矿"
                for decision in record.decisions
            )
            self.assertLessEqual(mines, 1)

    def test_equilibrium_policy_avoids_first_day_edge_collision(self) -> None:
        scenario = get_scenario(6)
        result = simulate_joint_policy(
            scenario,
            CongestionAwareEquilibriumPolicy(),
            (Weather.SUNNY,) * scenario.deadline,
        )
        first = result.records[1]
        self.assertEqual(first.states[0].water, 234)
        self.assertEqual(first.states[1].water, 234)
        self.assertEqual(first.states[2].water, 237)


if __name__ == "__main__":
    unittest.main()
