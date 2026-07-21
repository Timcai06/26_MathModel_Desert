import unittest

from desert.level5_game import best_response, evaluate_path, solve_level5_equilibrium
from desert.scenarios import get_scenario


class Level5GameTests(unittest.TestCase):
    def test_solo_best_response_is_feasible(self) -> None:
        scenario = get_scenario(5)
        best = best_response(scenario, None)
        replay = evaluate_path(scenario, best.path, None)
        self.assertEqual(best, replay)
        self.assertLessEqual(3 * best.initial_water + 2 * best.initial_food, 1200)

    def test_equilibrium_has_no_profitable_unilateral_deviation(self) -> None:
        equilibrium = solve_level5_equilibrium(get_scenario(5))
        self.assertEqual(equilibrium.player1_deviation_gain, 0)
        self.assertEqual(equilibrium.player2_deviation_gain, 0)


if __name__ == "__main__":
    unittest.main()
