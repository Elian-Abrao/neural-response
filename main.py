#!/usr/bin/env python3
"""Simple entry point for TRIBE v2 audio prediction."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from tribev2 import TribeModel


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
LATEST_DIR = OUTPUT_DIR / "latest"
HISTORY_DIR = OUTPUT_DIR / "history"
CACHE_DIR = ROOT / ".cache" / "tribev2-gpu-audio"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv()


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def find_input_audio() -> Path:
    candidates = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(
            "Nenhum audio encontrado em input/. Coloque um arquivo .wav, .mp3, .flac ou .ogg la."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Encontrei mais de um audio em input/: {names}. Deixe apenas um arquivo por vez."
        )
    return candidates[0]


def reset_latest_dir() -> None:
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    (LATEST_DIR / "figures").mkdir(parents=True, exist_ok=True)


def segment_to_record(segment) -> dict:
    return {
        "start": float(getattr(segment, "start", 0.0)),
        "duration": float(getattr(segment, "duration", 0.0)),
        "offset": float(getattr(segment, "offset", 0.0)),
        "timeline": getattr(segment, "timeline", None),
        "subject": getattr(segment, "subject", None),
        "event_count": len(getattr(segment, "ns_events", []) or []),
    }


def run_prediction(audio_path: Path) -> tuple[np.ndarray, list, pd.DataFrame]:
    config_update = {
        "data.text_feature.device": "cuda",
        "data.audio_feature.device": "cuda",
        "data.video_feature.image.device": "cuda",
    }
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(CACHE_DIR),
        device="cuda",
        config_update=config_update,
    )
    events = pd.DataFrame(
        [
            {
                "type": "Audio",
                "filepath": str(audio_path),
                "start": 0.0,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )
    preds, segments = model.predict(events=events, verbose=True)
    return preds, segments, events


def save_outputs(
    audio_path: Path, preds: np.ndarray, segments: list, events: pd.DataFrame
) -> None:
    segment_df = pd.DataFrame(segment_to_record(segment) for segment in segments)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio_path": str(audio_path),
        "prediction_shape": list(preds.shape),
        "kept_segments": len(segments),
        "prediction_dtype": str(preds.dtype),
        "prediction_min": float(preds.min()) if preds.size else None,
        "prediction_max": float(preds.max()) if preds.size else None,
        "prediction_mean": float(preds.mean()) if preds.size else None,
    }

    np.save(LATEST_DIR / "predictions.npy", preds)
    events.to_csv(LATEST_DIR / "events.csv", index=False)
    segment_df.to_csv(LATEST_DIR / "segments.csv", index=False)
    (LATEST_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = HISTORY_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["predictions.npy", "events.csv", "segments.csv", "summary.json"]:
        shutil.copy2(LATEST_DIR / name, run_dir / name)


def plot_segment_means(preds: np.ndarray) -> None:
    segment_means = preds.mean(axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(segment_means, marker="o")
    plt.title("Mean predicted response per segment")
    plt.xlabel("Segment index")
    plt.ylabel("Mean response")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LATEST_DIR / "figures" / "segment_means.png", dpi=180)
    plt.close()


def plot_vertex_heatmap(preds: np.ndarray) -> None:
    vertex_slice = preds[:, :256]
    plt.figure(figsize=(12, 4))
    plt.imshow(vertex_slice, aspect="auto", cmap="viridis")
    plt.colorbar(label="Predicted response")
    plt.title("Heatmap of the first 256 cortical vertices")
    plt.xlabel("Vertex index")
    plt.ylabel("Segment index")
    plt.tight_layout()
    plt.savefig(LATEST_DIR / "figures" / "vertex_heatmap.png", dpi=180)
    plt.close()


def plot_brain_map(values: np.ndarray, output_path: Path, title: str) -> None:
    fsavg = fetch_surf_fsaverage(mesh="fsaverage5")
    half = values.shape[0] // 2
    left = values[:half]
    right = values[half:]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        subplot_kw={"projection": "3d"},
    )
    plot_surf_stat_map(
        fsavg["infl_left"],
        left,
        hemi="left",
        view="lateral",
        colorbar=False,
        axes=axes[0],
        figure=fig,
        cmap="coolwarm",
    )
    axes[0].set_title("Left hemisphere")
    plot_surf_stat_map(
        fsavg["infl_right"],
        right,
        hemi="right",
        view="lateral",
        colorbar=True,
        axes=axes[1],
        figure=fig,
        cmap="coolwarm",
    )
    axes[1].set_title("Right hemisphere")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(audio_path: Path) -> None:
    report = f"""TRIBE v2 run completed.

Input audio:
- {audio_path}

Outputs:
- {LATEST_DIR / 'predictions.npy'}
- {LATEST_DIR / 'events.csv'}
- {LATEST_DIR / 'segments.csv'}
- {LATEST_DIR / 'summary.json'}

Figures:
- {LATEST_DIR / 'figures' / 'segment_means.png'}
- {LATEST_DIR / 'figures' / 'vertex_heatmap.png'}
- {LATEST_DIR / 'figures' / 'brain_first_segment.png'}
- {LATEST_DIR / 'figures' / 'brain_mean_response.png'}
"""
    (LATEST_DIR / "report.txt").write_text(report, encoding="utf-8")


def main() -> int:
    load_local_env()
    ensure_directories()
    reset_latest_dir()

    audio_path = find_input_audio()
    print(f"Using input file: {audio_path}")
    preds, segments, events = run_prediction(audio_path)
    save_outputs(audio_path, preds, segments, events)
    plot_segment_means(preds)
    plot_vertex_heatmap(preds)
    plot_brain_map(
        preds[0],
        LATEST_DIR / "figures" / "brain_first_segment.png",
        "TRIBE v2 brain map - first kept segment",
    )
    plot_brain_map(
        preds.mean(axis=0),
        LATEST_DIR / "figures" / "brain_mean_response.png",
        "TRIBE v2 brain map - mean response",
    )
    write_report(audio_path)

    print()
    print("Finished.")
    print(f"Results: {LATEST_DIR}")
    print(f"Summary: {LATEST_DIR / 'summary.json'}")
    print(f"Brain image: {LATEST_DIR / 'figures' / 'brain_mean_response.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
