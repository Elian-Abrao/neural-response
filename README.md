# neural-response

`neural-response` is a lightweight wrapper around Meta's
[TRIBE v2](https://github.com/facebookresearch/tribev2) for local brain-response
prediction experiments.

The project turns the original research code into a simpler workflow:

- drop one audio file into `input/`
- run `python main.py`
- inspect predictions and figures in `output/latest/`

## What it does

- loads the pretrained `facebook/tribev2` checkpoint
- runs prediction on a single audio input
- saves raw outputs as `npy`, `csv`, and `json`
- generates ready-to-open visualizations, including brain surface renders

## Why this repo is easier to use

- no CLI arguments required for the main workflow
- local `.env` support for `HF_TOKEN`
- fixed input/output folders
- automatic report and figure generation

## Quickstart

1. Activate the GPU environment:

```bash
source .venv-gpu/bin/activate
```

2. Put exactly one audio file in `input/`.

Supported formats:

- `.wav`
- `.mp3`
- `.flac`
- `.ogg`

3. Run:

```bash
python main.py
```

4. Open the results in `output/latest/`.

Important files:

- `output/latest/summary.json`
- `output/latest/predictions.npy`
- `output/latest/segments.csv`
- `output/latest/figures/brain_mean_response.png`
- `output/latest/figures/brain_first_segment.png`
- `output/latest/figures/segment_means.png`
- `output/latest/figures/vertex_heatmap.png`

## Notes

- The current simplified workflow is audio-first.
- TRIBE v2 itself supports text, audio, and video, but text/video still need
  extra environment work here because of the `whisperx` transcription path.
- Predictions are generated on the `fsaverage5` cortical surface.
