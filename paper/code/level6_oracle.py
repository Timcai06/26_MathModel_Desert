from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from .model import Action, DayDecision, Scenario, Weather
from .multiplayer import JointOnlinePolicy, MultiDayRecord


def _distances_to(scenario: Scenario, destination: int) -> dict[int, int]:
    distances = {destination: 0}
    queue = deque([destination])
    while queue:
        node = queue.popleft()
        for neighbour in scenario.graph[node]:
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def enumerate_shortest_routes(scenario: Scenario) -> tuple[tuple[int, ...], ...]:
    """枚举起点到终点的全部最短路，顺序固定以保证实验可复现。"""
    distances = _distances_to(scenario, scenario.destination)
    routes: list[tuple[int, ...]] = []

    def visit(node: int, prefix: tuple[int, ...]) -> None:
        if node == scenario.destination:
            routes.append(prefix)
            return
        for neighbour in sorted(scenario.graph[node]):
            if distances.get(neighbour) == distances[node] - 1:
                visit(neighbour, prefix + (neighbour,))

    visit(scenario.start, (scenario.start,))
    return tuple(routes)


def _route_step(route: tuple[int, ...], location: int) -> int:
    try:
        index = route.index(location)
    except ValueError as exc:
        raise RuntimeError(f"位置 {location} 不在偏离路线中") from exc
    if index + 1 >= len(route):
        return location
    return route[index + 1]


@dataclass(frozen=True)
class RouteProfilePolicy(JointOnlinePolicy):
    """第六关的异质最短路—装载—错峰策略剖面。"""

    routes: tuple[tuple[int, ...], ...]
    initial_loads: tuple[tuple[int, int], ...]
    voluntary_waits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.routes) == len(self.initial_loads) == len(self.voluntary_waits)
        ):
            raise ValueError("路线、装载与等待向量长度必须一致")
        if any(wait < 0 for wait in self.voluntary_waits):
            raise ValueError("自愿等待天数不能为负")

    def initial_purchases(self, scenario: Scenario) -> tuple[tuple[int, int], ...]:
        if len(self.initial_loads) != scenario.player_count:
            raise ValueError("策略人数与场景人数不一致")
        return self.initial_loads

    def decide(
        self,
        scenario: Scenario,
        history: tuple[MultiDayRecord, ...],
        weather: Weather,
    ) -> tuple[DayDecision | None, ...]:
        decisions: list[DayDecision | None] = []
        for player, state in enumerate(history[-1].states):
            if state.failed or state.arrival_day is not None:
                decisions.append(None)
                continue
            waits_used = sum(
                record.decisions[player] is not None
                and record.decisions[player].action is Action.STAY
                and record.weather is not Weather.SANDSTORM
                for record in history[1:]
            )
            if (
                state.location == scenario.start
                and waits_used < self.voluntary_waits[player]
                and weather is not Weather.SANDSTORM
            ):
                decisions.append(DayDecision(Action.STAY))
            elif weather is Weather.SANDSTORM:
                decisions.append(DayDecision(Action.STAY))
            else:
                decisions.append(
                    DayDecision(
                        Action.WALK,
                        _route_step(self.routes[player], state.location),
                    )
                )
        return tuple(decisions)

    def unilateral_deviation(
        self,
        player: int,
        *,
        route: tuple[int, ...] | None = None,
        initial_water: int | None = None,
        initial_food: int | None = None,
        voluntary_wait: int | None = None,
    ) -> "RouteProfilePolicy":
        routes = list(self.routes)
        loads = list(self.initial_loads)
        waits = list(self.voluntary_waits)
        if route is not None:
            routes[player] = route
        water, food = loads[player]
        loads[player] = (
            water if initial_water is None else initial_water,
            food if initial_food is None else initial_food,
        )
        if voluntary_wait is not None:
            waits[player] = voluntary_wait
        return replace(
            self,
            routes=tuple(routes),
            initial_loads=tuple(loads),
            voluntary_waits=tuple(waits),
        )
