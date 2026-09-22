"""Cross-session tracking: append each run's summary, chart drift over time."""
import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_FILENAME = "session_history.json"
TREND_FILENAME = "session_trend.png"


def _history_path(output_dir: str) -> str:
    return os.path.join(output_dir, HISTORY_FILENAME)


def append_session(output_dir: str, input_path: str, session_agg: dict, reps: list, timestamp: str = None) -> list:
    path = _history_path(output_dir)

    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)

    history.append({
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "input": os.path.basename(input_path),
        "n_reps": len(reps),
        "session_summary": session_agg,
    })

    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def save_trend_chart(history: list, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    if not history:
        ax.set_title("Session Trend")
        ax.text(0.5, 0.5, "no sessions yet", ha="center", va="center", transform=ax.transAxes)
    else:
        x = list(range(len(history)))
        metrics_present = set()
        for entry in history:
            metrics_present.update(entry.get("session_summary", {}).keys())

        for metric in sorted(metrics_present):
            xs, ys = [], []
            for xi, entry in zip(x, history):
                stats = entry.get("session_summary", {}).get(metric)
                if stats is not None:
                    xs.append(xi)
                    ys.append(stats["mean"])
            if xs:
                ax.plot(xs, ys, marker="o", label=metric)

        ax.set_xlabel("Session")
        ax.set_ylabel("Mean value")
        ax.set_title(f"Session Trend — {len(history)} session(s)")
        ax.set_xticks(x)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, TREND_FILENAME)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
