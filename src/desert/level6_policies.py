from __future__ import annotations

from dataclasses import dataclass

from .level4_policies import next_step, shortest_distance
from .model import Action, DayDecision, Scenario, Weather
from .multiplayer import JointOnlinePolicy, MultiDayRecord, PlayerState


INBOUND_ROUTES = (
    (1, 2, 3, 8, 13, 18),
    (1, 6, 7, 12, 17, 18),
    (1, 2, 3, 8, 13, 18),
)
OUTBOUND_ROUTES = (
    (18, 19, 24, 25),
    (18, 23, 24, 25),
    (18, 19, 24, 25),
)


def _route_step(route: tuple[int, ...], location: int) -> int:
    try:
        index = route.index(location)
    except ValueError as exc:
        raise RuntimeError(f"位置 {location} 不在角色路线中") from exc
    if index + 1 >= len(route):
        return location
    return route[index + 1]


def _action_count(history: tuple[MultiDayRecord, ...], player: int, action: Action) -> int:
    return sum(
        record.decisions[player] is not None
        and record.decisions[player].action is action
        for record in history[1:]
    )


def _observed_storms(history: tuple[MultiDayRecord, ...], today: Weather) -> int:
    return sum(record.weather is Weather.SANDSTORM for record in history[1:]) + int(
        today is Weather.SANDSTORM
    )


def _safe_after(
    scenario: Scenario,
    state: PlayerState,
    history: tuple[MultiDayRecord, ...],
    today: Weather,
    factor: int,
    storm_budget: int,
) -> bool:
    distance = shortest_distance(scenario, state.location, scenario.destination)
    storms_left = max(0, storm_budget - _observed_storms(history, today))
    water_after = state.water - factor * scenario.water.consumption[today]
    food_after = state.food - factor * scenario.food.consumption[today]
    required_water = 2 * distance * scenario.water.consumption[Weather.HOT] + (
        storms_left * scenario.water.consumption[Weather.SANDSTORM]
    )
    required_food = 2 * distance * scenario.food.consumption[Weather.HOT] + (
        storms_left * scenario.food.consumption[Weather.SANDSTORM]
    )
    remaining_days = scenario.deadline - (history[-1].day + 1)
    return (
        water_after >= required_water
        and food_after >= required_food
        and remaining_days >= distance + storms_left
    )


@dataclass(frozen=True)
class CoordinatedMinePolicy(JointOnlinePolicy):
    storm_budget: int = 5
    max_mines_per_player: int = 3

    def initial_purchases(self, scenario: Scenario) -> tuple[tuple[int, int], ...]:
        return ((240, 240),) * scenario.player_count

    def decide(
        self,
        scenario: Scenario,
        history: tuple[MultiDayRecord, ...],
        weather: Weather,
    ) -> tuple[DayDecision | None, ...]:
        states = history[-1].states
        decisions: list[DayDecision | None] = [None] * scenario.player_count
        mine_counts = [_action_count(history, i, Action.MINE) for i in range(scenario.player_count)]

        eligible: list[int] = []
        for i, state in enumerate(states):
            if (
                not state.failed
                and state.arrival_day is None
                and state.location in scenario.mines
                and mine_counts[i] < self.max_mines_per_player
                and _safe_after(scenario, state, history, weather, 3, self.storm_budget)
            ):
                eligible.append(i)
        preferred = (history[-1].day) % scenario.player_count
        assigned = preferred if preferred in eligible else (min(eligible) if eligible else None)

        for i, state in enumerate(states):
            if state.failed or state.arrival_day is not None:
                continue
            if state.location == scenario.start and i == 2:
                voluntary_waits = sum(
                    record.decisions[i] is not None
                    and record.decisions[i].action is Action.STAY
                    and record.weather is not Weather.SANDSTORM
                    for record in history[1:]
                )
                if voluntary_waits < 1 and weather is not Weather.SANDSTORM:
                    decisions[i] = DayDecision(Action.STAY)
                    continue
            if state.location in scenario.mines:
                if assigned == i:
                    decisions[i] = DayDecision(Action.MINE)
                elif weather is Weather.SANDSTORM:
                    decisions[i] = DayDecision(Action.STAY)
                elif mine_counts[i] >= self.max_mines_per_player or not _safe_after(
                    scenario, state, history, weather, 1, self.storm_budget
                ):
                    decisions[i] = DayDecision(
                        Action.WALK, _route_step(OUTBOUND_ROUTES[i], state.location)
                    )
                else:
                    decisions[i] = DayDecision(Action.STAY)
                continue
            visited_mine = any(
                record.states[i].location in scenario.mines for record in history
            )
            if weather is Weather.SANDSTORM:
                decisions[i] = DayDecision(Action.STAY)
            elif visited_mine:
                decisions[i] = DayDecision(
                    Action.WALK, _route_step(OUTBOUND_ROUTES[i], state.location)
                )
            else:
                decisions[i] = DayDecision(
                    Action.WALK, _route_step(INBOUND_ROUTES[i], state.location)
                )
        return tuple(decisions)


@dataclass(frozen=True)
class CongestionAwareEquilibriumPolicy(JointOnlinePolicy):
    """异路错峰行走，并在矿山采用逐日加入博弈的纯策略均衡。

    第六关任一天加入已有 ``k`` 人矿组的边际收益为 ``B/(k+1)``，
    相比停留多消耗两倍基础资源。三种天气下第三名玩家的名义边际收益
    仍非负，因此所有资源安全的在场玩家都会挖矿。
    """

    storm_budget: int = 5

    def initial_purchases(self, scenario: Scenario) -> tuple[tuple[int, int], ...]:
        return ((240, 240),) * scenario.player_count

    def decide(self, scenario, history, weather):
        decisions: list[DayDecision | None] = []
        for i, state in enumerate(history[-1].states):
            if state.failed or state.arrival_day is not None:
                decisions.append(None)
                continue
            if state.location == scenario.start and i == 2:
                voluntary_waits = sum(
                    record.decisions[i] is not None
                    and record.decisions[i].action is Action.STAY
                    and record.weather is not Weather.SANDSTORM
                    for record in history[1:]
                )
                if voluntary_waits < 1 and weather is not Weather.SANDSTORM:
                    decisions.append(DayDecision(Action.STAY))
                    continue
            if state.location in scenario.mines:
                if _safe_after(scenario, state, history, weather, 3, self.storm_budget):
                    decisions.append(DayDecision(Action.MINE))
                elif weather is Weather.SANDSTORM:
                    decisions.append(DayDecision(Action.STAY))
                else:
                    decisions.append(
                        DayDecision(Action.WALK, _route_step(OUTBOUND_ROUTES[i], state.location))
                    )
                continue
            visited_mine = any(
                record.states[i].location in scenario.mines for record in history
            )
            if weather is Weather.SANDSTORM:
                decisions.append(DayDecision(Action.STAY))
            elif visited_mine:
                decisions.append(
                    DayDecision(Action.WALK, _route_step(OUTBOUND_ROUTES[i], state.location))
                )
            else:
                decisions.append(
                    DayDecision(Action.WALK, _route_step(INBOUND_ROUTES[i], state.location))
                )
        return tuple(decisions)


@dataclass(frozen=True)
class CoordinatedDirectPolicy(JointOnlinePolicy):
    initial_boxes: int = 185

    routes = (
        (1, 2, 3, 4, 5, 10, 15, 20, 25),
        (1, 6, 11, 16, 21, 22, 23, 24, 25),
        (1, 2, 3, 4, 5, 10, 15, 20, 25),
    )

    def initial_purchases(self, scenario: Scenario) -> tuple[tuple[int, int], ...]:
        return ((self.initial_boxes, self.initial_boxes),) * scenario.player_count

    def decide(self, scenario, history, weather):
        decisions: list[DayDecision | None] = []
        for i, state in enumerate(history[-1].states):
            if state.failed or state.arrival_day is not None:
                decisions.append(None)
                continue
            if state.location == scenario.start and i == 2:
                waits = sum(
                    record.decisions[i] is not None
                    and record.decisions[i].action is Action.STAY
                    and record.weather is not Weather.SANDSTORM
                    for record in history[1:]
                )
                if waits < 1 and weather is not Weather.SANDSTORM:
                    decisions.append(DayDecision(Action.STAY))
                    continue
            if weather is Weather.SANDSTORM:
                decisions.append(DayDecision(Action.STAY))
            else:
                decisions.append(DayDecision(Action.WALK, _route_step(self.routes[i], state.location)))
        return tuple(decisions)


@dataclass(frozen=True)
class MirrorMinePolicy(JointOnlinePolicy):
    storm_budget: int = 5

    def initial_purchases(self, scenario: Scenario) -> tuple[tuple[int, int], ...]:
        return ((240, 240),) * scenario.player_count

    def decide(self, scenario, history, weather):
        decisions: list[DayDecision | None] = []
        for i, state in enumerate(history[-1].states):
            if state.failed or state.arrival_day is not None:
                decisions.append(None)
                continue
            visited_mine = any(record.states[i].location in scenario.mines for record in history)
            if state.location in scenario.mines and _safe_after(
                scenario, state, history, weather, 3, self.storm_budget
            ):
                decisions.append(DayDecision(Action.MINE))
            elif weather is Weather.SANDSTORM:
                decisions.append(DayDecision(Action.STAY))
            else:
                target = scenario.destination if visited_mine else min(scenario.mines)
                decisions.append(DayDecision(Action.WALK, next_step(scenario, state.location, target)))
        return tuple(decisions)
