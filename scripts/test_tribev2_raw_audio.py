#!/usr/bin/env python3
"""Run TRIBE v2 on a raw audio event without transcription."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tribev2 import TribeModel


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRIBE v2 with a manual Audio event, skipping whisperx."
    )
    parser.add_argument("--audio", required=True, help="Path to a wav/mp3/flac/ogg file.")
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--cache-dir", default=".cache/tribev2-raw-audio")
    parser.add_argument("--device", default="cpu", help='Use "cpu" or "cuda".')
    parser.add_argument(
        "--output-dir",
        default="outputs/tribev2",
        help="Directory where predictions and metadata will be written.",
    )
    return parser.parse_args()


def segment_to_record(segment) -> dict:
    return {
        "start": float(getattr(segment, "start", 0.0)),
        "duration": float(getattr(segment, "duration", 0.0)),
        "offset": float(getattr(segment, "offset", 0.0)),
        "timeline": getattr(segment, "timeline", None),
        "subject": getattr(segment, "subject", None),
        "event_count": len(getattr(segment, "ns_events", []) or []),
    }


def main() -> int:
    load_local_env()
    args = parse_args()
    audio = Path(args.audio).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_update = {}
    if args.device in {"cpu", "cuda"}:
        config_update = {
            "data.text_feature.device": args.device,
            "data.audio_feature.device": args.device,
            "data.video_feature.image.device": args.device,
        }

    model = TribeModel.from_pretrained(
        args.checkpoint,
        cache_folder=args.cache_dir,
        device=args.device,
        config_update=config_update,
    )

    events = pd.DataFrame(
        [
            {
                "type": "Audio",
                "filepath": str(audio),
                "start": 0.0,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )

    print(f"Running TRIBE v2 on audio={audio}")
    preds, segments = model.predict(events=events, verbose=True)
    summary = {
        "audio_path": str(audio),
        "device": args.device,
        "checkpoint": args.checkpoint,
        "prediction_shape": list(preds.shape),
        "kept_segments": len(segments),
        "prediction_dtype": str(preds.dtype),
        "prediction_min": float(preds.min()) if preds.size else None,
        "prediction_max": float(preds.max()) if preds.size else None,
        "prediction_mean": float(preds.mean()) if preds.size else None,
    }
    segment_records = [segment_to_record(segment) for segment in segments]
    segment_df = pd.DataFrame(segment_records)

    np.save(output_dir / "predictions.npy", preds)
    events.to_csv(output_dir / "events.csv", index=False)
    segment_df.to_csv(output_dir / "segments.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Prediction array shape: {preds.shape}")
    print(f"Number of kept segments: {len(segments)}")
    print(f"Saved predictions to {output_dir / 'predictions.npy'}")
    print(f"Saved segments to {output_dir / 'segments.csv'}")
    print(f"Saved summary to {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
