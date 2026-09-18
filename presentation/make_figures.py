"""Generate the two daily-frequency charts used by the Beamer deck."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "crsp_monthly_walkforward_2024"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["reversal", "ridge", "tree"]
COLORS = {"reversal": "#D55E00", "ridge": "#0072B2", "tree": "#009E73"}
STYLES = {"reversal": "--", "ridge": "-", "tree": "-."}
MARKERS = {"reversal": "o", "ridge": "s", "tree": "^"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#102b4c", "text.color": "#102b4c",
                     "axes.edgecolor": "#7f9bb9", "xtick.color": "#3e6083", "ytick.color": "#3e6083"})


def save(name):
    ax = plt.gca()
    ax.grid(axis="y", color="#e6f0fa", linewidth=.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


fig, ax = plt.subplots(figsize=(8, 3.1))
for model in MODELS:
    d = pd.read_csv(OUT / f"ic_{model}.csv", parse_dates=["date"]).sort_values("date")
    ax.scatter(d.date, d.rank_ic, s=10, alpha=.18, color=COLORS[model], label="_nolegend_")
    ax.plot(d.date, d.rank_ic.rolling(10, min_periods=10).mean(), lw=2, ls=STYLES[model],
            label=f"{model.title()} · 10-day mean", color=COLORS[model])
ax.axhline(0, color="#a6bdd5", lw=.8)
ax.set_ylabel("Daily Rank IC")
ax.set_xlabel("Trading day, Jul–Nov 2024")
ax.legend(frameon=False, ncol=1, loc="lower right", fontsize=8)
save("daily_rank_ic.pdf")

common = set(pd.read_csv(OUT / "common_portfolio_dates.csv").date)
fig, ax = plt.subplots(figsize=(8, 3.1))
for model in MODELS:
    d = pd.read_csv(OUT / f"portfolio_{model}.csv", parse_dates=["date"])
    d = d.loc[d.date.dt.strftime("%Y-%m-%d").isin(common)].sort_values("date")
    assert len(d) == len(common) and d.net.notna().all()
    ax.plot(d.date, (1+d.net).cumprod()-1, label=model.title(), color=COLORS[model],
            lw=2, ls=STYLES[model], marker=MARKERS[model], markersize=3,
            markevery=6)
ax.axhline(0, color="#a6bdd5", lw=.8)
ax.set_ylabel("Cumulative net return")
ax.set_xlabel(f"Observed trading day · {len(common)} common dates")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
ax.legend(frameon=False, ncol=3, loc="lower left")
save("net_portfolio_common.pdf")

print("Generated two daily-frequency figures in", FIG)
