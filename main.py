"""CLI entry point for the shot form analyzer."""
import argparse
import logging
import os
import sys

from analyzer.pose import PoseExtractor
from analyzer.segmenter import segment_reps

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

    return frames, fps, reps_bounds, args


if __name__ == "__main__":
    main()
