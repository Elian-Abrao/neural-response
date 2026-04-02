#!/usr/bin/env python3
"""Minimal smoke test for Meta's TRIBE v2 model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load TRIBE v2 and optionally run inference on one input."
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/tribev2",
        help="Local checkpoint directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/tribev2",
        help="Directory used by TRIBE v2 to cache extracted features.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device forwarded to TribeModel.from_pretrained. Use "cpu", "cuda", or "auto".',
    )
    parser.add_argument(
        "--text",
        help="Path to a .txt file to test text inference.",
    )
    parser.add_argument(
        "--audio",
        help="Path to an audio file to test audio inference.",
    )
    parser.add_argument(
        "--video",
        help="Path to a video file to test video inference.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load the pretrained model without running inference.",
    )
    return parser.parse_args()


def resolve_single_input(args: argparse.Namespace) -> tuple[str, str] | None:
    provided = [
        ("text_path", args.text),
        ("audio_path", args.audio),
        ("video_path", args.video),
    ]
    provided = [(name, value) for name, value in provided if value]
    if args.load_only:
        if provided:
            raise SystemExit("--load-only cannot be combined with --text/--audio/--video")
        return None
    if len(provided) != 1:
        raise SystemExit(
            "Provide exactly one of --text, --audio, or --video, or use --load-only."
        )
    return provided[0]


def main() -> int:
    load_local_env()
    args = parse_args()
    source = resolve_single_input(args)

    try:
        from tribev2 import TribeModel
    except Exception as exc:  # pragma: no cover
        print("Failed to import tribev2.", file=sys.stderr)
        print(
            "Install it first, for example:",
            file=sys.stderr,
        )
        print(
            '  pip install -e "git+https://github.com/facebookresearch/tribev2.git#egg=tribev2"',
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_update = {}
    if args.device in {"cpu", "cuda"}:
        config_update = {
            "data.text_feature.device": args.device,
            "data.audio_feature.device": args.device,
            "data.video_feature.image.device": args.device,
        }

    print(f"Loading TRIBE v2 from {args.checkpoint} on device={args.device}...")
    model = TribeModel.from_pretrained(
        args.checkpoint,
        cache_folder=str(cache_dir),
        device=args.device,
        config_update=config_update,
    )
    print("Model loaded successfully.")

    if source is None:
        return 0

    key, value = source
    path = Path(value).expanduser().resolve()
    print(f"Building events dataframe from {key}={path}")

    try:
        events = model.get_events_dataframe(**{key: str(path)})
    except FileNotFoundError as exc:
        if "uvx" in str(exc):
            print(
                "TRIBE v2 tried to call `uvx whisperx` to transcribe audio/text, "
                "but `uvx` is not installed in this environment.",
                file=sys.stderr,
            )
            print(
                "Install uv/uvx first, or provide a workflow that already has "
                "word-level transcripts cached.",
                file=sys.stderr,
            )
        raise
    print(f"Generated events dataframe with shape={events.shape}")
    print(f"Event columns: {list(events.columns)}")

    try:
        preds, segments = model.predict(events=events, verbose=True)
    except OSError as exc:
        message = str(exc)
        if "gated repo" in message.lower() or "meta-llama/Llama-3.2-3B" in message:
            print(
                "Prediction reached the text feature extractor, but Hugging Face "
                "blocked access to `meta-llama/Llama-3.2-3B`.",
                file=sys.stderr,
            )
            print(
                "You likely need an authenticated HF session plus explicit access "
                "to that gated model before text inference will work.",
                file=sys.stderr,
            )
        raise
    print(f"Prediction array shape: {preds.shape}")
    print(f"Number of kept segments: {len(segments)}")

    if preds.size:
        print(
            "Prediction summary:",
            {
                "dtype": str(preds.dtype),
                "min": float(preds.min()),
                "max": float(preds.max()),
                "mean": float(preds.mean()),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
