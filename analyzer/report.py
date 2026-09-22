"""Session report artifacts: JSON + CSV + matplotlib summary chart."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = (
    ("elbow_at_release", "Elbow Angle at Release (deg)"),
    ("knee_bend_at_setup", "Knee Bend at Setup (deg)"),
    ("wrist_follow_through", "Wrist Follow-Through (deg)"),
    ("arc_proxy", "Arc Proxy — Shoulder Elevation at Release (deg)"),
)

CSV_FIELDS = (
    "rep_id",
    "start_frame",
    "release_frame",
    "end_frame",
    "elbow_at_release",
    "knee_bend_at_setup",
    "wrist_follow_through",
    "arc_proxy",
    "elbow_flare",
    "confidence",
    "low_confidence",
    "out_of_range",
    "label",
)


def save_csv(reps: list, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "session_report.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rep in reps:
            row = dict(rep)
            row["out_of_range"] = ";".join(row.get("out_of_range") or [])
            writer.writerow(row)
    return path


def save_json(session_agg: dict, reps: list, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "session_report.json")
    with open(path, "w") as f:
        json.dump({"reps": reps, "session_summary": session_agg}, f, indent=2)
    return path


def _plot_panel(ax, reps, field, title):
    if not reps:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no reps detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Rep")
        ax.set_ylabel(title)
        return

    rep_ids = [rep["rep_id"] for rep in reps]
    values = [rep[field] for rep in reps]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    ax.bar(rep_ids, values, color="#4C72B0")
    ax.axhline(mean, color="black", linestyle="--", linewidth=1, label="mean")
    ax.axhspan(mean - std, mean + std, color="gray", alpha=0.2, label="±1σ")
    ax.set_title(title)
    ax.set_xlabel("Rep")
    ax.set_ylabel(title)
    ax.set_xticks(rep_ids)
    ax.legend(fontsize=8)


def save_chart(reps: list, session_agg: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Session Summary — {len(reps)} reps")

    for ax, (field, title) in zip(axes.flat, PANELS):
        _plot_panel(ax, reps, field, title)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "session_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
