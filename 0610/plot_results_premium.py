import csv
import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "ab_decode_priority_scheduler_raw_data.csv"
OUTPUT_PNG = BASE_DIR / "ab_decode_priority_scheduler_premium.png"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


def parse_result(file_path):
    if not file_path.exists():
        return None
    content = file_path.read_text()

    fields = {
        "throughput_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "total_token_throughput_tok_s": r"Total token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
    }

    row = {}
    for key, pattern in fields.items():
        match = re.search(pattern, content)
        row[key] = float(match.group(1)) if match else None

    if row["mean_ttft_ms"] is None or row["mean_tpot_ms"] is None:
        return None
    return row


def discover_concurrencies(*folders):
    values = set()
    pattern = re.compile(r"_c(\d+)\.txt$")
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            match = pattern.search(path.name)
            if match:
                values.add(int(match.group(1)))
    return sorted(values)


def pct_change(new, old):
    if new is None or old in (None, 0):
        return None
    return (new - old) / old * 100.0


series = {
    "A_OFF": {
        "folder": BASE_DIR / "A_OFF_test",
        "filename": "result_OFF_test_c{c}.txt",
        "label": "A/OFF: stock vLLM scheduler",
        "color": "#4b5563",
        "marker": "o",
        "ls": "--",
        "data": {},
    },
    "B_ON": {
        "folder": BASE_DIR / "B_ON_test",
        "filename": "result_ON_test_c{c}.txt",
        "label": "B/ON: QAIC decode-priority scheduler",
        "color": "#0ea5e9",
        "marker": "D",
        "ls": "-",
        "data": {},
    },
}


concurrencies = discover_concurrencies(*(s["folder"] for s in series.values()))

for name, config in series.items():
    for c in concurrencies:
        result = parse_result(config["folder"] / config["filename"].format(c=c))
        if result is not None:
            config["data"][c] = result


with OUTPUT_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "concurrency",
        "off_mean_ttft_ms",
        "on_mean_ttft_ms",
        "ttft_delta_on_minus_off_ms",
        "ttft_delta_pct",
        "off_mean_tpot_ms",
        "on_mean_tpot_ms",
        "tpot_delta_on_minus_off_ms",
        "tpot_delta_pct",
        "tpot_improvement_pct_positive_is_better",
        "off_output_token_throughput_tok_s",
        "on_output_token_throughput_tok_s",
        "throughput_delta_pct",
        "off_p99_ttft_ms",
        "on_p99_ttft_ms",
        "off_p99_tpot_ms",
        "on_p99_tpot_ms",
    ])

    for c in concurrencies:
        off = series["A_OFF"]["data"].get(c, {})
        on = series["B_ON"]["data"].get(c, {})

        off_ttft = off.get("mean_ttft_ms")
        on_ttft = on.get("mean_ttft_ms")
        off_tpot = off.get("mean_tpot_ms")
        on_tpot = on.get("mean_tpot_ms")
        off_tps = off.get("throughput_tok_s")
        on_tps = on.get("throughput_tok_s")

        tpot_delta_pct = pct_change(on_tpot, off_tpot)
        writer.writerow([
            c,
            "" if off_ttft is None else f"{off_ttft:.2f}",
            "" if on_ttft is None else f"{on_ttft:.2f}",
            "" if off_ttft is None or on_ttft is None else f"{on_ttft - off_ttft:.2f}",
            "" if pct_change(on_ttft, off_ttft) is None else f"{pct_change(on_ttft, off_ttft):.2f}",
            "" if off_tpot is None else f"{off_tpot:.2f}",
            "" if on_tpot is None else f"{on_tpot:.2f}",
            "" if off_tpot is None or on_tpot is None else f"{on_tpot - off_tpot:.2f}",
            "" if tpot_delta_pct is None else f"{tpot_delta_pct:.2f}",
            "" if tpot_delta_pct is None else f"{-tpot_delta_pct:.2f}",
            "" if off_tps is None else f"{off_tps:.2f}",
            "" if on_tps is None else f"{on_tps:.2f}",
            "" if pct_change(on_tps, off_tps) is None else f"{pct_change(on_tps, off_tps):.2f}",
            "" if off.get("p99_ttft_ms") is None else f"{off['p99_ttft_ms']:.2f}",
            "" if on.get("p99_ttft_ms") is None else f"{on['p99_ttft_ms']:.2f}",
            "" if off.get("p99_tpot_ms") is None else f"{off['p99_tpot_ms']:.2f}",
            "" if on.get("p99_tpot_ms") is None else f"{on['p99_tpot_ms']:.2f}",
        ])


try:
    import matplotlib.pyplot as plt
except ImportError:
    print(f"Raw data saved to {OUTPUT_CSV}")
    print("matplotlib is not installed; skipped plot generation.")
    raise SystemExit(0)


def values_for(key, metric):
    xs = []
    ys = []
    for c in concurrencies:
        row = series[key]["data"].get(c)
        if row is None or row.get(metric) is None:
            continue
        xs.append(c)
        ys.append(row[metric])
    return xs, ys


def plot_by_concurrency(ax, metric, ylabel, title):
    for key, config in series.items():
        xs, ys = values_for(key, metric)
        ax.plot(
            xs,
            ys,
            config["ls"],
            marker=config["marker"],
            color=config["color"],
            linewidth=2.5,
            markersize=7,
            label=config["label"],
        )
        for x, y in zip(xs, ys):
            ax.annotate(f"c={x}", (x, y), textcoords="offset points",
                        xytext=(4, 7), fontsize=8, color=config["color"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="-", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_tradeoff(ax, x_metric, y_metric, xlabel, ylabel, title, invert_x=False):
    for key, config in series.items():
        xs = []
        ys = []
        labels = []
        for c in concurrencies:
            row = config["data"].get(c)
            if row is None or row.get(x_metric) is None or row.get(y_metric) is None:
                continue
            xs.append(row[x_metric])
            ys.append(row[y_metric])
            labels.append(c)
        ax.plot(
            xs,
            ys,
            config["ls"],
            color=config["color"],
            linewidth=2.5,
            alpha=0.85,
            label=config["label"],
        )
        ax.scatter(xs, ys, marker=config["marker"], color=config["color"], s=70, zorder=5)
        for x, y, c in zip(xs, ys, labels):
            ax.annotate(f"c={c}", (x, y), textcoords="offset points",
                        xytext=(4, 7), fontsize=8, color=config["color"])
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="-", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(16, 11))

plot_by_concurrency(
    axes[0, 0],
    "mean_ttft_ms",
    "Mean TTFT (ms)",
    "TTFT vs Concurrency",
)
plot_by_concurrency(
    axes[0, 1],
    "mean_tpot_ms",
    "Mean TPOT (ms)",
    "TPOT vs Concurrency",
)
plot_tradeoff(
    axes[1, 0],
    "mean_ttft_ms",
    "throughput_tok_s",
    "Mean TTFT (ms)",
    "Output token throughput (tok/s)",
    "Throughput / TTFT Tradeoff",
    invert_x=True,
)
plot_tradeoff(
    axes[1, 1],
    "mean_tpot_ms",
    "throughput_tok_s",
    "Mean TPOT (ms)",
    "Output token throughput (tok/s)",
    "Throughput / TPOT Tradeoff",
    invert_x=True,
)

fig.suptitle(
    "QAIC Decode-Priority Scheduler A/B Test\n"
    "A/OFF = stock mixed prefill/decode scheduler, B/ON = decode-priority mitigation\n"
    "Qwen2.5-7B-Instruct, max-model-len=2048, max-num-seqs=96",
    fontsize=17,
    fontweight="bold",
    y=0.985,
)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.895),
    ncol=2,
    frameon=True,
    fontsize=11,
)
fig.text(
    0.5,
    0.015,
    "Benchmark by RAG dataset. Lower TTFT/TPOT is better; higher throughput is better.",
    ha="center",
    fontsize=10,
    color="#6b7280",
)

plt.tight_layout(rect=(0, 0.04, 1, 0.82))
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

print(f"Raw data saved to {OUTPUT_CSV}")
print(f"A/B plot saved to {OUTPUT_PNG}")
