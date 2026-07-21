from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables" / "generated"


def main() -> None:
    data = json.loads((ROOT / "output/problem1/known_weather_exact.json").read_text())
    lines = [
        r"\begin{longtable}{ccclrrrr}",
        r"\caption{第一、二关全局最优计划的逐日审计流水}\label{tab:daily-ledger}\\",
        r"\toprule",
        r"关卡 & 日 & 天气 & 位置/行动 & 现金 & 水 & 食物 & 购买(水/食)\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{\small 表\thetable\ （续）}\\",
        r"\toprule",
        r"关卡 & 日 & 天气 & 位置/行动 & 现金 & 水 & 食物 & 购买(水/食)\\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{8}{r}{续下页}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for level in data:
        for record in level["records"]:
            weather = record["weather"] or "--"
            action = record["action"] or "初始"
            movement = f"{record['location']}/{action}"
            purchase = f"{record['buy_water']}/{record['buy_food']}"
            lines.append(
                f"{level['level']} & {record['day']} & {weather} & {movement} & "
                f"{record['cash']} & {record['water']} & {record['food']} & {purchase}\\\\"
            )
    lines.append(r"\end{longtable}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "known_weather_daily.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "known_weather_daily.tex")


if __name__ == "__main__":
    main()
