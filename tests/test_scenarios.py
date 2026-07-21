import unittest

from desert.scenarios import SCENARIOS


class ScenarioTests(unittest.TestCase):
    def test_all_graphs_are_complete_symmetric_and_connected(self) -> None:
        for scenario in SCENARIOS.values():
            self.assertEqual(set(scenario.graph), set(range(1, scenario.node_count + 1)))
            reached = {scenario.start}
            frontier = [scenario.start]
            while frontier:
                node = frontier.pop()
                for neighbour in scenario.graph[node]:
                    self.assertIn(node, scenario.graph[neighbour])
                    if neighbour not in reached:
                        reached.add(neighbour)
                        frontier.append(neighbour)
            self.assertEqual(len(reached), scenario.node_count)

    def test_key_level_2_hex_edges(self) -> None:
        graph = SCENARIOS[2].graph
        for left, right in ((13, 22), (22, 30), (30, 39), (39, 46), (46, 55), (56, 64)):
            self.assertIn(right, graph[left])

    def test_level_4_is_four_neighbour_grid(self) -> None:
        graph = SCENARIOS[4].graph
        self.assertEqual(graph[1], frozenset({2, 6}))
        self.assertEqual(graph[13], frozenset({8, 12, 14, 18}))
        self.assertEqual(graph[25], frozenset({20, 24}))


if __name__ == "__main__":
    unittest.main()

