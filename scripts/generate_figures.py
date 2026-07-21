from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd

from desert.scenarios import get_scenario


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "candidate_figures"
PAPER_OUT = ROOT / "paper" / "figures" / "generated"

# 黑白印刷专用调色板。语义区分依靠线型、标记和纹理，而非色相。
NAVY = "#111111"
TEAL = "#3A3A3A"
GOLD = "#737373"
RED = "#202020"
SAND = "#F1F1F1"
INK = "#111111"
GRAY = "#626262"
LIGHT = "#D1D1D1"

FINAL_FIGURES = {
    "maps_overview", "model_workflow", "known_weather_ledger", "level3_mdp",
    "level4_frontier", "level5_equilibrium", "level6_tradeoff",
    "level6_mechanism", "evidence_matrix", "oos_risk_intervals",
    "loading_robustness", "role_fairness",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#8C8C8C",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#3F3F3F",
            "ytick.color": "#3F3F3F",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "font.size": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    if name in FINAL_FIGURES:
        fig.savefig(PAPER_OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(PAPER_OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _clean(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#E3E3E3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def maps_overview() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    specs = [(1, "第一关：27区不规则图"), (2, "第二关：8×8六边形图"),
             (3, "第三/五关：13区图"), (4, "第四/六关：5×5网格")]
    known = json.loads((ROOT / "output/problem1/known_weather_exact.json").read_text())
    known_paths = {
        row["level"]: [record["location"] for record in row["records"]]
        for row in known
    }
    for ax, (level, title) in zip(axes.flat, specs):
        scenario = get_scenario(level)
        graph = nx.Graph()
        for node, neighbours in scenario.graph.items():
            for neighbour in neighbours:
                graph.add_edge(node, neighbour)
        if level == 2:
            pos = {node: (((node - 1) % 8) + 0.5 * (((node - 1) // 8) % 2),
                          -((node - 1) // 8)) for node in scenario.graph}
        elif level == 4:
            pos = {node: ((node - 1) % 5, -((node - 1) // 5)) for node in scenario.graph}
        else:
            pos = nx.spring_layout(graph, seed=26, k=0.72 if level == 1 else 0.9)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#C9C9C9", width=0.65)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="white", edgecolors=NAVY,
                               linewidths=0.7, node_size=95 if level != 2 else 42)
        if level in known_paths:
            path_edges = list(zip(known_paths[level], known_paths[level][1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, ax=ax, edge_color=NAVY,
                                   width=2.3, arrows=False)
        special = {scenario.start: ("#FFFFFF", "S"), scenario.destination: ("#111111", "T")}
        special.update({node: ("#B0B0B0", "V") for node in scenario.villages})
        special.update({node: ("#4A4A4A", "M") for node in scenario.mines})
        for node, (color, label) in special.items():
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], ax=ax, node_color=color,
                                   edgecolors=NAVY, linewidths=1.0,
                                   node_size=175 if level != 2 else 90)
            label_color = "white" if color in {"#111111", "#4A4A4A"} else INK
            ax.text(*pos[node], label, ha="center", va="center", color=label_color,
                    fontsize=7 if level != 2 else 5.5, fontweight="bold")
        ax.set_title(title, loc="left", pad=8)
        ax.axis("off")
    fig.suptitle("六关地图归并为四种图结构", fontsize=15, fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.935, "S起点 · T终点 · V村庄 · M矿山；粗黑线为第一、二关最优轨迹",
             color=GRAY, fontsize=9)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.91])
    save(fig, "maps_overview")


def model_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.axis("off")
    columns = [
        (0.04, "信息集", [("全时段天气已知", "E1/E3"), ("仅知当天天气", "E2/E4"), ("观察他人历史", "E3/E4")]),
        (0.36, "模型", [("时间扩展网络 MILP", "全局最优"), ("有限时域 MDP", "反馈策略"), ("动态博弈/联合模拟", "拥挤响应")]),
        (0.68, "验证", [("HiGHS证书+独立回放", "逐日合法"), ("全序列/样本外", "风险区间"), ("最佳响应+外部性消融", "偏离增益")]),
    ]
    colors = [NAVY, TEAL, RED]
    for x, header, rows in columns:
        ax.text(x, 0.84, header, transform=ax.transAxes, fontsize=12, fontweight="bold", color=NAVY)
        for i, (main, sub) in enumerate(rows):
            y = 0.64 - i * 0.23
            box = FancyBboxPatch((x, y), 0.25, 0.15, boxstyle="round,pad=0.012,rounding_size=0.018",
                                 transform=ax.transAxes, facecolor="white", edgecolor=colors[i], linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + 0.018, y + 0.095, main, transform=ax.transAxes, fontsize=9.5, fontweight="semibold")
            ax.text(x + 0.018, y + 0.045, sub, transform=ax.transAxes, fontsize=8, color=GRAY)
    for y in (0.715, 0.485, 0.255):
        ax.annotate("", xy=(0.665, y), xytext=(0.615, y), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color=GOLD))
        ax.annotate("", xy=(0.345, y), xytext=(0.295, y), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color=GOLD))
    ax.text(0.04, 0.985, "从确定性最优到多人反馈：统一状态—决策—证据链",
            transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0.04, 0.025, "证据等级随信息条件变化：不把样本外推荐误称为无条件全局最优",
            transform=ax.transAxes, fontsize=9, color=GRAY)
    save(fig, "model_workflow")


def known_weather_ledger() -> None:
    data = json.loads((ROOT / "output/problem1/known_weather_exact.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.6), sharex=False)
    weather_colors = {"晴朗": "#F7F7F7", "高温": "#E7E7E7", "沙暴": "#CECECE"}
    for ax, level in zip(axes, data):
        frame = pd.DataFrame(level["records"])
        for row in frame.iloc[1:].itertuples():
            ax.axvspan(row.day - 0.48, row.day + 0.48, color=weather_colors[row.weather], zorder=0)
        ax.plot(frame.day, frame.water, color=TEAL, marker="o", ms=2.8, lw=1.8, label="水/箱")
        ax.plot(frame.day, frame.food, color=GOLD, marker="s", ms=2.5, lw=1.6,
                linestyle="--", label="食物/箱")
        ax2 = ax.twinx()
        ax2.step(frame.day, frame.cash, color=NAVY, lw=1.7, where="post", label="现金/元")
        for row in frame.itertuples():
            if row.action == "挖矿":
                ax.scatter(row.day, row.water, marker="^", s=35, color=NAVY, zorder=5)
            if row.buy_water or row.buy_food:
                ax.scatter(row.day, row.water, marker="D", s=34, color=RED, zorder=5)
        ax.set_ylabel("库存/箱")
        ax2.set_ylabel("现金/元", color=NAVY)
        ax.set_title(f"第{level['level']}关 · 最优终值 {level['final_value']:.0f} 元 · 第 {level['arrival_day']} 天到达",
                     loc="left")
        _clean(ax)
        ax2.spines["top"].set_visible(False)
        ax.set_xlim(-0.5, max(frame.day) + 0.5)
    axes[-1].set_xlabel("日期")
    handles = [Line2D([0], [0], color=TEAL, marker="o", label="水"),
               Line2D([0], [0], color=GOLD, ls="--", marker="s", label="食物"),
               Line2D([0], [0], color=NAVY, label="现金"),
               Line2D([0], [0], color=NAVY, marker="^", lw=0, label="挖矿"),
               Line2D([0], [0], color=RED, marker="D", lw=0, label="补给")]
    fig.legend(handles=handles, ncol=5, loc="upper right", bbox_to_anchor=(0.96, 0.965))
    fig.suptitle("确定天气精确解的逐日资源—现金审计", x=0.07, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.07, 0.93, "背景带：晴朗 / 高温 / 沙暴；所有轨迹均由独立模拟器回放",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0.04, 0.03, 0.97, 0.9])
    save(fig, "known_weather_ledger")


def level3_mdp() -> None:
    df = pd.read_csv(ROOT / "output/problem2/level3_mdp_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    x = df.hot_probability
    ax.fill_between(x, df.minimum_value, df.maximum_value, color=GOLD, alpha=0.18,
                    label="支撑内最小—最大")
    ax.plot(x, df.expected_final_value, color=NAVY, lw=2.2, marker="o", label="期望终值")
    ax.plot(x, df.p05_value, color=TEAL, lw=1.7, marker="s", ls="--", label="5%分位")
    for row in df.itertuples():
        ax.annotate(f"{row.expected_final_value:.0f}", (row.hot_probability, row.expected_final_value),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("高温概率")
    ax.set_ylabel("终值/元")
    ax.set_xticks(x, [f"{v:.0%}" for v in x])
    fig.suptitle("第三关：高温概率升高只压低价值，不改变54/54直达策略",
                 x=0.10, y=0.98, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.10, 0.91, "每个概率点精确枚举1024条天气序列，失败数均为0",
             color=GRAY, fontsize=8.5)
    _clean(ax)
    ax.legend(loc="lower left")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.86])
    save(fig, "level3_mdp")


def level4_frontier() -> None:
    df = pd.read_csv(ROOT / "output/problem2/level4_parameter_search.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=False)
    for ax, prefix, title in [(axes[0], "baseline", "基准天气"), (axes[1], "adverse", "不利天气")]:
        x = df[f"{prefix}_failure_rate"] * 100
        y = df[f"{prefix}_mean_penalized_value"]
        scatter = ax.scatter(x, y, c=df.sandstorm_budget, cmap="Greys", s=18, alpha=0.72,
                             edgecolor="#555555", linewidth=0.2)
        chosen = df[(df.sandstorm_budget == 5) & (df.initial_water == 240) & (df.initial_food == 240)].iloc[0]
        ax.scatter(chosen[f"{prefix}_failure_rate"] * 100,
                   chosen[f"{prefix}_mean_penalized_value"], marker="*", s=170,
                   color=NAVY, edgecolor="white", linewidth=0.8, zorder=5)
        ax.axvline(0.2 if prefix == "baseline" else 0.5, color=GRAY, ls="--", lw=1.0)
        ax.set_xlabel("训练失败率/%")
        ax.set_ylabel("总体终值均值/元")
        ax.set_title(title, loc="left")
        _clean(ax)
    cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", shrink=0.38,
                        fraction=0.05, pad=0.18, aspect=30)
    cbar.set_label("沙暴预算/天")
    fig.suptitle("第四关567个候选的收益—失败风险前沿", x=0.06, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.06, 0.885, "黑星：240/240箱、B5；虚线：预先设定风险预算；失败按0元计入均值",
             color=GRAY, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.25, wspace=0.25)
    save(fig, "level4_frontier")


def level5_equilibrium() -> None:
    payload = json.loads((ROOT / "output/problem3/summary.json").read_text())["level5"]
    scenario = get_scenario(5)
    graph = nx.Graph()
    for node, neighbours in scenario.graph.items():
        for neighbour in neighbours:
            graph.add_edge(node, neighbour)
    pos = nx.spring_layout(graph, seed=26, k=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#D6D6D6", width=0.8)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="white", edgecolors=NAVY, node_size=260)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)
    paths = [[1] + [a["destination"] for a in payload[p]["actions"] if a["destination"]]
             for p in ("player1", "player2")]
    for path, color, style in zip(paths, (TEAL, RED), ("solid", "dashed")):
        nx.draw_networkx_edges(graph, pos, edgelist=list(zip(path, path[1:])), ax=ax,
                               edge_color=color, width=3, style=style)
    ax.set_title("空间路径：同日有向边不重合", loc="left")
    ax.axis("off")
    ax = axes[1]
    labels = ["玩家1", "玩家2"]
    colors = [TEAL, RED]
    for row, (key, label, color) in enumerate(zip(("player1", "player2"), labels, colors)):
        actions = payload[key]["actions"]
        loc = 1
        for day, action in enumerate(actions, start=1):
            text = "停留" if action["action"] == "停留" else f"{loc}→{action['destination']}"
            ax.barh(row, 0.88, left=day - 0.44, height=0.42, color=color,
                    alpha=0.9 if action["action"] != "停留" else 0.28,
                    edgecolor=color, linewidth=1)
            ax.text(day, row, text, ha="center", va="center", fontsize=8,
                    color="white" if action["action"] != "停留" else INK)
            if action["destination"]:
                loc = action["destination"]
    ax.set_yticks([0, 1], labels)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("日期")
    ax.set_xlim(0.45, 4.55)
    ax.set_title("时序均衡：9535元 vs 9510元", loc="left")
    _clean(ax, "x")
    ax.text(0.02, -0.22, "完整最佳响应复核：两名玩家最大单边偏离增益均为0元",
            transform=ax.transAxes, color=GRAY, fontsize=8.5)
    fig.suptitle("第五关非对称纯策略 Nash 均衡", x=0.06, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0.05, 0.98, 0.9])
    save(fig, "level5_equilibrium")


def level6_tradeoff() -> None:
    df = pd.read_csv(ROOT / "output/problem3/level6_oos_comparison.csv")
    order = ["镜像挖矿", "拥挤感知非合作", "合作轮换", "分路直达"]
    colors = {"镜像挖矿": "#858585", "拥挤感知非合作": "#626262",
              "合作轮换": "#3E3E3E", "分路直达": "#111111"}
    linestyles = {"镜像挖矿": ":", "拥挤感知非合作": "--",
                  "合作轮换": "-.", "分路直达": "-"}
    markers = {"低沙暴": "o", "基准": "s", "不利": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"width_ratios": [1.0, 1.25]})
    for panel, ax in enumerate(axes):
        panel_order = order if panel == 0 else order[1:]
        for policy in panel_order:
            sub = df[df.policy == policy]
            ax.plot(sub.player_failure_rate * 100, sub.mean_individual_value,
                    color=colors[policy], lw=1.7, ls=linestyles[policy], alpha=0.9)
            for row in sub.itertuples():
                ax.scatter(row.player_failure_rate * 100, row.mean_individual_value,
                           color=colors[policy], marker=markers[row.weather_model], s=58,
                           edgecolor="white", linewidth=0.7, zorder=4)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("玩家失败率/%")
        ax.set_title("全策略视图" if panel == 0 else "非镜像策略放大", loc="left")
        _clean(ax)
    axes[0].set_ylabel("总体个体终值均值/元")
    axes[1].set_ylim(6000, 9000)
    policy_handles = [Line2D([0], [0], color=colors[p], ls=linestyles[p], lw=2, label=p) for p in order]
    weather_handles = [Line2D([0], [0], marker=m, color=INK, lw=0, label=w)
                       for w, m in markers.items()]
    fig.legend(handles=policy_handles, loc="lower left", bbox_to_anchor=(0.07, 0.015), ncol=4)
    fig.legend(handles=weather_handles, loc="lower right", bbox_to_anchor=(0.96, 0.015), ncol=3)
    fig.suptitle("第六关：分路直达是唯一跨分布满足风险预算的策略",
                 x=0.06, y=0.98, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.91, "每点20000条天气、60000个玩家观测；失败终值记为0；横轴为对称对数尺度",
             color=GRAY, fontsize=8.5)
    fig.tight_layout(rect=[0.03, 0.12, 0.99, 0.84], w_pad=2.2)
    save(fig, "level6_tradeoff")


def level6_mechanism() -> None:
    df = pd.read_csv(ROOT / "output/problem3/level6_oos_comparison.csv")
    df = df[df.weather_model == "基准"].set_index("policy").loc[
        ["镜像挖矿", "拥挤感知非合作", "合作轮换", "分路直达"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3))
    metrics = [
        ("shared_edge_player_days_per_session", "共享边玩家日/局", NAVY),
        ("shared_mine_player_days_per_session", "共享矿山玩家日/局", GOLD),
        ("player_failure_rate", "玩家失败率/%", RED),
    ]
    for ax, (column, title, color) in zip(axes, metrics):
        values = df[column] * (100 if column == "player_failure_rate" else 1)
        bars = ax.barh(df.index, values, color=color, alpha=0.88, edgecolor="white")
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xlim(0, max(values) * 1.18 + 1e-6)
        _clean(ax, "x")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("第六关失效机制：镜像策略把单人路径放大为共享边灾难", x=0.05,
                 y=0.98, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.05, 0.90, "基准天气，20000局；分路直达的共享边与共享矿山玩家日均为0",
             color=GRAY, fontsize=9)
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.80], w_pad=2.2)
    save(fig, "level6_mechanism")


def evidence_matrix() -> None:
    rows = ["第一关", "第二关", "第三关", "第四关", "第五关", "第六关"]
    evidence = [1, 1, 2, 4, 3, 4]
    results = ["10470元", "12730元", "54/54直达", "B5：10163元", "9535/9510元", "直达：7768元"]
    labels = {1: "E1\n最优证书", 2: "E2\n全枚举", 3: "E3\n偏离复核", 4: "E4\n样本外"}
    colors = {1: NAVY, 2: TEAL, 3: GOLD, 4: RED}
    fig, ax = plt.subplots(figsize=(9.3, 4.6))
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    for i, (row, e, result) in enumerate(zip(rows, evidence, results)):
        y = 5 - i
        ax.add_patch(FancyBboxPatch((0.05, y - 0.32), 4.85, 0.64,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor="#FAFAFA" if i % 2 == 0 else "white",
                                    edgecolor="#E5E5E5", linewidth=0.7))
        ax.text(0.22, y, row, va="center", fontweight="semibold")
        ax.text(1.35, y, result, va="center", color=INK)
        ax.add_patch(FancyBboxPatch((3.48, y - 0.22), 1.12, 0.44,
                                    boxstyle="round,pad=0.015,rounding_size=0.08",
                                    facecolor=colors[e], edgecolor="none"))
        ax.text(4.04, y, labels[e], ha="center", va="center", color="white", fontsize=8)
    ax.text(0.05, 5.72, "关卡", fontweight="bold", color=GRAY)
    ax.text(1.35, 5.72, "核心结果", fontweight="bold", color=GRAY)
    ax.text(3.48, 5.72, "证据等级", fontweight="bold", color=GRAY)
    ax.set_title("六关结论与证据等级总览", loc="left", fontsize=15, pad=20)
    ax.text(0.05, -0.7, "E1/E2为精确结论；E3为完整最佳响应检验；E4为预注册风险预算下的独立样本外推荐",
            color=GRAY, fontsize=8.5)
    fig.tight_layout()
    save(fig, "evidence_matrix")


def oos_risk_intervals() -> None:
    """把点估计、Wilson区间和事先给定的风险预算放在同一坐标系。"""
    level4 = pd.read_csv(ROOT / "output/problem2/level4_oos_validation.csv")
    level4 = level4[level4.policy == "网格搜索入选"].copy()
    level6 = pd.read_csv(ROOT / "output/problem3/level6_oos_comparison.csv")
    level6 = level6[level6.policy == "分路直达"].copy()
    weather_order = ["低沙暴", "基准", "不利"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    panels = [
        (axes[0], level4, "failure_rate", "failure_wilson_low", "failure_wilson_high",
         "第四关 · B5阈值策略", 20000),
        (axes[1], level6, "player_failure_rate", "player_failure_wilson_low",
         "player_failure_wilson_high", "第六关 · 分路直达", 60000),
    ]
    for ax, frame, value, low, high, title, observations in panels:
        frame = frame.set_index("weather_model").loc[weather_order]
        y = frame[value].to_numpy() * 100
        lower = np.maximum(0.0, (frame[value] - frame[low]).to_numpy(dtype=float) * 100)
        upper = np.maximum(0.0, (frame[high] - frame[value]).to_numpy(dtype=float) * 100)
        x = np.arange(3)
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", color=INK,
                    ecolor="#555555", elinewidth=1.4, capsize=4, ms=5.5, zorder=4)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}%", (xi, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.plot([-0.35, 1.35], [0.2, 0.2], color="#555555", ls="--", lw=1.0)
        ax.plot([1.65, 2.35], [0.5, 0.5], color="#111111", ls=":", lw=1.3)
        ax.set_xticks(x, weather_order)
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.96, f"每种天气 {observations:,} 个观测",
                transform=ax.transAxes, va="top", color=GRAY, fontsize=8)
        _clean(ax)
    axes[0].set_ylabel("失败率及95% Wilson区间/%")
    axes[0].set_ylim(-0.04, 0.62)
    fig.suptitle("样本外风险点估计与上置信界同时通过预算", x=0.06, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.06, 0.895, "虚线为基准0.2%预算，点线为不利0.5%预算；低沙暴仅作敏感性参照",
             color=GRAY, fontsize=8.8)
    fig.tight_layout(rect=[0.04, 0.03, 0.99, 0.84], w_pad=2.0)
    save(fig, "oos_risk_intervals")


def loading_robustness() -> None:
    """展示第六关初始装载对风险和收益的单调权衡。"""
    df = pd.read_csv(ROOT / "output/problem3/level6_load_search.csv")
    x = df.initial_water
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.plot(x, df.baseline_failure_rate * 100, color=INK, marker="o", ms=4,
            lw=1.8, label="基准")
    ax.plot(x, df.adverse_failure_rate * 100, color="#666666", marker="s", ms=4,
            lw=1.6, ls="--", label="不利")
    ax.axhline(0.2, color="#444444", ls="--", lw=0.9)
    ax.axhline(0.5, color="#111111", ls=":", lw=1.1)
    ax.axvline(185, color="#777777", ls="-.", lw=1.0)
    ax.scatter([185], [df.loc[df.initial_water == 185, "adverse_failure_rate"].iloc[0] * 100],
               marker="*", s=120, color=INK, edgecolor="white", zorder=5)
    ax.set_xlabel("每名玩家初始水/食物箱数（等量）")
    ax.set_ylabel("训练失败率/%")
    ax.set_title("风险随装载增加快速下降", loc="left")
    ax.legend()
    _clean(ax)

    ax = axes[1]
    ax.plot(x, df.baseline_mean_value, color=INK, marker="o", ms=4, lw=1.8, label="基准均值")
    ax.plot(x, df.adverse_mean_value, color="#666666", marker="s", ms=4,
            lw=1.6, ls="--", label="不利均值")
    ax.fill_between(x, df.adverse_p05, df.adverse_mean_value, color="#D8D8D8",
                    alpha=0.7, label="不利5%分位至均值")
    ax.axvline(185, color="#777777", ls="-.", lw=1.0)
    ax.scatter([185], [df.loc[df.initial_water == 185, "adverse_mean_value"].iloc[0]],
               marker="*", s=120, color=INK, edgecolor="white", zorder=5)
    ax.set_xlabel("每名玩家初始水/食物箱数（等量）")
    ax.set_ylabel("总体个体终值/元")
    ax.set_title("额外安全库存以终值下降为代价", loc="left")
    ax.legend(loc="lower left")
    _clean(ax)
    fig.suptitle("第六关装载敏感性：185/185是风险约束下的最小安全装载",
                 x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.895, "训练集每点3000局；黑星为入选装载；失败样本终值按0计入",
             color=GRAY, fontsize=8.8)
    fig.tight_layout(rect=[0.04, 0.03, 0.99, 0.84], w_pad=2.0)
    save(fig, "loading_robustness")


def role_fairness() -> None:
    """比较固定角色收益，并显示随机轮换后的事前公平值。"""
    df = pd.read_csv(ROOT / "output/problem3/level6_oos_comparison.csv")
    df = df[df.policy == "分路直达"].set_index("weather_model").loc[["低沙暴", "基准", "不利"]]
    roles = ["role1_mean", "role2_mean", "role3_mean"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=True)
    for ax, (weather, row) in zip(axes, df.iterrows()):
        values = [row[col] for col in roles]
        mean = float(np.mean(values))
        ax.plot([1, 2, 3], values, color="#666666", ls="--", lw=1.1)
        for i, (value, marker) in enumerate(zip(values, markers), start=1):
            ax.scatter(i, value, marker=marker, s=58, facecolor="white",
                       edgecolor=INK, linewidth=1.3, zorder=4)
        ax.axhline(mean, color=INK, lw=1.5, label="角色随机化后的事前均值")
        ax.set_xticks([1, 2, 3], ["角色1", "角色2", "角色3"])
        ax.set_title(weather, loc="left")
        ax.text(0.04, 0.07, f"角色差={row.role_mean_gap:.1f}元\n事前均值={mean:.1f}元",
                transform=ax.transAxes, fontsize=8, color=GRAY)
        _clean(ax)
    axes[0].set_ylabel("固定角色样本外均值/元")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("固定路线存在角色差异，出发前等概率轮换实现事前公平",
                 x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.895, "三条路线的空间长度与沙暴暴露不同；随机置换不改变拥挤与安全性",
             color=GRAY, fontsize=8.8)
    fig.tight_layout(rect=[0.04, 0.03, 0.99, 0.84], w_pad=1.6)
    save(fig, "role_fairness")


def main() -> None:
    setup()
    maps_overview()
    model_workflow()
    known_weather_ledger()
    level3_mdp()
    level4_frontier()
    level5_equilibrium()
    level6_tradeoff()
    level6_mechanism()
    evidence_matrix()
    oos_risk_intervals()
    loading_robustness()
    role_fairness()
    print(OUT)


if __name__ == "__main__":
    main()
