from __future__ import annotations

import json
from pathlib import Path

from desert.level4_mdp import Level4FixedLoadMDP
from desert.model import Weather
from desert.scenarios import get_scenario
from desert.weather import IIDWeatherModel


OUTPUT = Path("output/problem2/level4_exact_mdp_gap.json")


def main() -> None:
    scenario = get_scenario(4)
    model = IIDWeatherModel(
        {Weather.SUNNY: 0.45, Weather.HOT: 0.45, Weather.SANDSTORM: 0.10}
    )
    risk_budget = 0.002
    failure_penalty = 79800.0
    benchmark = Level4FixedLoadMDP(
        scenario,
        model,
        initial_water=240,
        initial_food=240,
        failure_penalty=failure_penalty,
    ).solve(5)
    # 对任意 λ≥0：V*_{P(fail)<=α} ≤ maxπ E[J-λI_fail]+λα。
    dual_upper = benchmark.optimal.expected_final_value + failure_penalty * risk_budget
    payload = {
        "scope": {
            "weather": "基准IID：晴朗0.45、高温0.45、沙暴0.10",
            "initial_load": [benchmark.initial_water, benchmark.initial_food],
            "purchases": "禁止；与240/240满负重阈值策略的实际运行条件一致",
            "action_space": "完整5×5地图上的移动、停留、挖矿；当天气揭示后决策",
            "interpretation": "区间只适用于固定装载、无补给子问题，不外推为原第四关全局最优区间",
        },
        "risk_constrained_dual": {
            "risk_budget": risk_budget,
            "failure_penalty": failure_penalty,
            "lagrangian_policy_expected_final_value": benchmark.optimal.expected_final_value,
            "lagrangian_policy_failure_probability": benchmark.optimal.failure_probability,
            "dual_upper_bound": dual_upper,
            "states_evaluated": benchmark.optimal_states,
        },
        "threshold_B5": {
            "expected_final_value": benchmark.threshold.expected_final_value,
            "failure_probability": benchmark.threshold.failure_probability,
            "states_evaluated": benchmark.threshold_states,
            "risk_feasible": benchmark.threshold.failure_probability <= risk_budget,
        },
        "certified_interval": {
            "feasible_lower_bound": benchmark.threshold.expected_final_value,
            "dual_upper_bound": dual_upper,
            "absolute_width": dual_upper - benchmark.threshold.expected_final_value,
            "relative_width_to_upper": (dual_upper - benchmark.threshold.expected_final_value) / dual_upper,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
