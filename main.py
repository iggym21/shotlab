"""CLI entry point for the shot form analyzer."""
import argparse
import logging
import os
import sys

import cv2

from analyzer.pose import PoseExtractor
from analyzer.segmenter import segment_reps
from analyzer.metrics import compute_rep_metrics, aggregate_session
from analyzer.classifier import train_and_label
from analyzer.report import save_json, save_chart, save_csv
from analyzer.history import append_session, save_trend_chart
from analyzer.overlay import draw_frame

logger = logging.getLogger("shotlab")


def _build_frame_rep_lookup(reps):
    """frame_idx -> (rep_id, in_rep, out_of_range) for every frame covered by any rep."""
    lookup = {}
    for rep in reps:
        for idx in range(rep["start_frame"], rep["end_frame"] + 1):
            lookup[idx] = (rep["rep_id"], True, rep.get("out_of_range", []))
    return lookup


def render_annotated_video(input_path: str, output_dir: str, frames: list, reps: list, fps: float) -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not reopen video for rendering: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    basename = os.path.basename(input_path)
    name, _ = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"annotated_{name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_rep_lookup = _build_frame_rep_lookup(reps)

    frame_idx = 0
    while True:
        ok, raw_frame = cap.read()
        if not ok:
            break

        frame_data = frames[frame_idx] if frame_idx < len(frames) else None
        rep_id, in_rep, out_of_range = frame_rep_lookup.get(frame_idx, (None, False, None))
        timestamp_s = frame_idx / fps

        annotated = draw_frame(raw_frame, frame_data, rep_id, timestamp_s, in_rep, out_of_range=out_of_range)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze basketball free-throw form from video.")
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", default="output/", help="Output directory (default: output/).")
    parser.add_argument("--fps", type=float, default=None, help="Override detected video fps.")
    parser.add_argument("--min-reps", type=int, default=3, help="Minimum reps required to run the classifier.")
    parser.add_argument(
        "--shooting-side", choices=("right", "left"), default="right",
        help="Which arm is the shooting arm (default: right).",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Skip annotated video, stats only.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not os.path.isfile(args.input):
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    frames, detected_fps = PoseExtractor(shooting_side=args.shooting_side).extract(args.input)
    fps = args.fps or detected_fps
    print(f"Extracted {len(frames)} frames at {fps:.2f} fps.")

    if not any(frame is not None for frame in frames):
        logger.error("No usable pose landmarks detected in any frame of %s.", args.input)
        sys.exit(1)

    reps_bounds = segment_reps(frames, fps, shooting_side=args.shooting_side)
    print(f"Detected {len(reps_bounds)} rep(s):")
    for i, (start, release, end) in enumerate(reps_bounds):
        print(
            f"  rep {i}: start={start} ({start/fps:.2f}s) "
            f"release={release} ({release/fps:.2f}s) "
            f"end={end} ({end/fps:.2f}s)"
        )

    if not reps_bounds:
        logger.warning("No reps detected — skipping metrics, classifier, and report.")
        return frames, fps, [], {}, args

    reps = compute_rep_metrics(reps_bounds, frames, shooting_side=args.shooting_side)
    if len(reps) < args.min_reps:
        logger.info(
            "Only %d rep(s) detected (< --min-reps=%d); classifier will label all reps 'unclassified'.",
            len(reps), args.min_reps,
        )
    reps = train_and_label(reps, output_dir=args.output, min_reps=args.min_reps)
    session_agg = aggregate_session(reps)

    print(
        f"\n{'rep_id':>6} | {'elbow':>7} | {'knee':>7} | {'wrist':>7} | {'arc':>7} | "
        f"{'flare':>6} | {'conf':>5} | label | flags"
    )
    for rep in reps:
        flags = ",".join(rep["out_of_range"]) if rep["out_of_range"] else "-"
        if rep["low_confidence"]:
            flags = f"{flags},low_confidence" if flags != "-" else "low_confidence"
        print(
            f"{rep['rep_id']:>6} | {rep['elbow_at_release']:>7.1f} | "
            f"{rep['knee_bend_at_setup']:>7.1f} | {rep['wrist_follow_through']:>7.1f} | "
            f"{rep['arc_proxy']:>7.1f} | {rep['elbow_flare']:>6.2f} | {rep['confidence']:>5.2f} | "
            f"{rep['label']} | {flags}"
        )

    json_path = save_json(session_agg, reps, args.output)
    csv_path = save_csv(reps, args.output)
    chart_path = save_chart(reps, session_agg, args.output)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {chart_path}")

    history = append_session(args.output, args.input, session_agg, reps)
    trend_path = save_trend_chart(history, args.output)
    print(f"Wrote {trend_path}")

    if not args.no_overlay:
        video_path = render_annotated_video(args.input, args.output, frames, reps, fps)
        print(f"Wrote {video_path}")

    return frames, fps, reps, session_agg, args


if __name__ == "__main__":
    main()
