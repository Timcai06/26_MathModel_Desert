import unittest

from desert.model import Action, DayDecision, Plan, Weather
from desert.reference_plans import LEVEL_1_REFERENCE, LEVEL_2_REFERENCE
from desert.scenarios import get_scenario
from desert.simulator import SimulationError, simulate


class SimulatorTests(unittest.TestCase):
    def test_level_1_reference_ledger(self) -> None:
        result = simulate(get_scenario(1), LEVEL_1_REFERENCE)
        self.assertEqual(result.arrival_day, 24)
        self.assertEqual(result.final_value, 10470)
        self.assertEqual(
            (result.records[-1].location, result.records[-1].cash, result.records[-1].water, result.records[-1].food),
            (27, 10470, 0, 0),
        )
        self.assertEqual(
            (result.records[8].cash, result.records[8].water, result.records[8].food),
            (4150, 243, 235),
        )

    def test_level_2_reference_ledger(self) -> None:
        result = simulate(get_scenario(2), LEVEL_2_REFERENCE)
        self.assertEqual(result.arrival_day, 30)
        self.assertEqual(result.final_value, 12730)
        self.assertEqual(
            (result.records[19].cash, result.records[19].water, result.records[19].food),
            (5730, 196, 200),
        )

    def test_walking_during_sandstorm_is_rejected(self) -> None:
        plan = Plan(100, 100, tuple([DayDecision(Action.STAY)] * 3 + [DayDecision(Action.WALK, 2)]))
        with self.assertRaisesRegex(SimulationError, "沙暴"):
            simulate(get_scenario(1), plan)

    def test_mining_at_non_mine_is_rejected(self) -> None:
        with self.assertRaisesRegex(SimulationError, "不是矿山"):
            simulate(get_scenario(1), Plan(100, 100, (DayDecision(Action.MINE),)))

    def test_overweight_initial_purchase_is_rejected(self) -> None:
        with self.assertRaisesRegex(SimulationError, "负重"):
            simulate(get_scenario(1), Plan(401, 0, ()))

    def test_purchase_before_action_at_village(self) -> None:
        plan = Plan(
            100,
            100,
            (
                DayDecision(Action.WALK, 2),
                DayDecision(Action.WALK, 3),
                DayDecision(Action.WALK, 4),
                DayDecision(Action.WALK, 9),
                DayDecision(Action.WALK, 14),
                DayDecision(Action.WALK, 15, buy_before_water=1, buy_before_food=1),
            ),
        )
        weather = (Weather.SUNNY,) * 30
        result = simulate(get_scenario(4), plan, weather)
        self.assertEqual((result.records[-1].location, result.records[-1].water, result.records[-1].food), (15, 65, 53))


if __name__ == "__main__":
    unittest.main()
