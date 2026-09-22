"""CLI entry point for the shot form analyzer."""
import argparse
import logging
import os
import sys

from analyzer.pose import PoseExtractor
from analyzer.segmenter import segment_reps
from analyzer.metrics import compute_rep_metrics, aggregate_session
from analyzer.classifier import train_and_label

logger = logging.getLogger("shotlab")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze basketball free-throw form from video.")
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", default="output/", help="Output directory (default: output/).")
    parser.add_argument("--fps", type=float, default=None, help="Override detected video fps.")
    parser.add_argument("--min-reps", type=int, default=3, help="Minimum reps required to run the classifier.")
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

    frames, detected_fps = PoseExtractor().extract(args.input)
    fps = args.fps or detected_fps
    print(f"Extracted {len(frames)} frames at {fps:.2f} fps.")

    if not any(frame is not None for frame in frames):
        logger.error("No usable pose landmarks detected in any frame of %s.", args.input)
        sys.exit(1)

    reps_bounds = segment_reps(frames, fps)
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

    reps = compute_rep_metrics(reps_bounds, frames)
    if len(reps) < args.min_reps:
        logger.info(
            "Only %d rep(s) detected (< --min-reps=%d); classifier will label all reps 'unclassified'.",
            len(reps), args.min_reps,
        )
    reps = train_and_label(reps, output_dir=args.output, min_reps=args.min_reps)
    session_agg = aggregate_session(reps)

    print(f"\n{'rep_id':>6} | {'elbow':>7} | {'knee':>7} | {'wrist':>7} | {'arc':>7} | label")
    for rep in reps:
        print(
            f"{rep['rep_id']:>6} | {rep['elbow_at_release']:>7.1f} | "
            f"{rep['knee_bend_at_setup']:>7.1f} | {rep['wrist_follow_through']:>7.1f} | "
            f"{rep['arc_proxy']:>7.1f} | {rep['label']}"
        )

    return frames, fps, reps, session_agg, args


if __name__ == "__main__":
    main()
