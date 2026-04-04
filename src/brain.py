"""
Inference and analysis layer for the Ad Brain Analyzer.
Wraps tribev2 with caching, statistics, and brain map rendering.
"""

import hashlib
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

CACHE_DIR = Path("outputs/cache")
RENDER_DIR = Path("outputs/renders")

# HCP ROI groups used for marketing-relevant score breakdowns
VISUAL_ROIS = ["V1", "V2", "V3", "V4", "MT", "MST", "V3A", "V3B"]
LANGUAGE_ROIS = ["44", "45", "STSdp", "STSda", "STSvp", "STSva", "TE1a", "TE1m"]
ATTENTION_ROIS = ["FEF", "7PC", "VIP", "LIPv", "LIPd", "IPS1"]


def hash_video(video_path: str) -> str:
    """SHA256 hash of a video file, used as cache key."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA256 hash of in-memory bytes, used as cache key for API uploads."""
    return hashlib.sha256(data).hexdigest()


def _patch_whisperx_compute_type():
    """
    tribev2 hardcodes compute_type='float16' for whisperx, which fails on CPU.
    Monkey-patch it to use 'int8' when running without a GPU.
    """
    import json
    import os
    import subprocess
    import tempfile

    import torch
    from tribev2.eventstransforms import ExtractWordsFromAudio

    @staticmethod  # type: ignore[misc]
    def _patched(wav_filename, language):
        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [
                "uvx", "whisperx",
                str(wav_filename),
                "--model", "large-v3",
                "--language", language_codes[language],
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", "8",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]
            cmd = [c for c in cmd if c]
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"].replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                words.append({
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                })
        return __import__("pandas").DataFrame(words)

    ExtractWordsFromAudio._get_transcript_from_audio = _patched


def run_inference(video_path: str, model) -> tuple[np.ndarray, list]:
    """
    Run TRIBE v2 inference on a video file.
    Results are cached on disk by file hash — repeat calls return instantly.

    Returns (preds, segments) where preds is shape (n_segments, n_vertices).
    """
    _patch_whisperx_compute_type()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vid_hash = hash_video(video_path)
    cache_path = CACHE_DIR / f"{vid_hash}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["preds"], data["segments"]

    events_df = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=events_df)

    with open(cache_path, "wb") as f:
        pickle.dump({"preds": preds, "segments": segments}, f)

    return preds, segments


def _safe_roi_mean(preds_mean: np.ndarray, roi_names: list[str]) -> float:
    """Average activation across a list of ROIs, silently skipping missing ones."""
    from tribev2.utils import get_hcp_roi_indices

    values = []
    for roi in roi_names:
        try:
            indices = get_hcp_roi_indices(roi, hemi="both", mesh="fsaverage5")
            values.append(float(preds_mean[indices].mean()))
        except (ValueError, KeyError):
            pass
    return float(np.mean(values)) if values else 0.0


def compute_stats(preds: np.ndarray, segments: list) -> dict:
    """
    Derive marketing-relevant statistics from raw model predictions.

    Returns a dict with:
      - overall_score: mean BOLD activation across all vertices and segments
      - peak_segment_idx: index of segment with highest mean activation
      - peak_timestamp_s: time (seconds) of peak segment
      - engagement_over_time: np.ndarray (n_segments,) — per-TR mean activation
      - segment_timestamps: list[float] — start time of each segment
      - top_rois: list[str] — top 10 activated brain regions (HCP labels)
      - roi_scores: dict[str, float] — all ROIs → mean activation
      - visual_score: float — visual cortex activation
      - language_score: float — language cortex activation
      - attention_score: float — attention network activation
    """
    from tribev2.utils import get_hcp_labels, get_topk_rois, summarize_by_roi

    preds_mean = preds.mean(axis=0)  # (n_vertices,)
    engagement_over_time = preds.mean(axis=1)  # (n_segments,)

    peak_idx = int(np.argmax(engagement_over_time))
    try:
        timestamps = [float(seg.start) for seg in segments]
    except Exception:
        timestamps = [i * 2.0 for i in range(len(segments))]

    peak_ts = timestamps[peak_idx] if timestamps else 0.0

    # get_topk_rois passes dict_keys directly to np.array(), which creates a
    # 0-d object array on some numpy versions. Re-implement with list() fix.
    roi_labels = list(get_hcp_labels(mesh="fsaverage5").keys())
    roi_summary = summarize_by_roi(preds_mean)
    top_k_idx = np.argsort(roi_summary)[::-1][:10]
    top_rois = [roi_labels[i] for i in top_k_idx]

    roi_values = summarize_by_roi(preds_mean)
    roi_names = list(get_hcp_labels(mesh="fsaverage5").keys())
    roi_scores = dict(zip(roi_names, [float(v) for v in roi_values]))

    return {
        "overall_score": float(preds.mean()),
        "peak_segment_idx": peak_idx,
        "peak_timestamp_s": float(peak_ts),
        "engagement_over_time": engagement_over_time,
        "segment_timestamps": timestamps,
        "top_rois": top_rois,
        "roi_scores": roi_scores,
        "visual_score": _safe_roi_mean(preds_mean, VISUAL_ROIS),
        "language_score": _safe_roi_mean(preds_mean, LANGUAGE_ROIS),
        "attention_score": _safe_roi_mean(preds_mean, ATTENTION_ROIS),
        "preds_mean": preds_mean,
    }


def compute_delta_stats(stats_a: dict, stats_b: dict) -> dict:
    """
    Compare two ads. Returns winner and per-ROI differences.
    """
    score_a = stats_a["overall_score"]
    score_b = stats_b["overall_score"]
    delta = score_b - score_a

    if abs(delta) < 1e-6:
        winner = "tie"
    else:
        winner = "B" if delta > 0 else "A"

    # ROIs where the ads differ most
    shared_rois = set(stats_a["roi_scores"]) & set(stats_b["roi_scores"])
    roi_deltas = {
        roi: stats_b["roi_scores"][roi] - stats_a["roi_scores"][roi]
        for roi in shared_rois
    }
    top_diff = sorted(roi_deltas, key=lambda r: abs(roi_deltas[r]), reverse=True)[:10]

    return {
        "winner": winner,
        "score_delta": float(delta),
        "roi_deltas": roi_deltas,
        "top_differentiating_rois": top_diff,
    }


def render_brain_png(
    preds_mean: np.ndarray,
    output_path: str,
    vmax: float | None = None,
    cmap: str = "hot",
    symmetric_cbar: bool = False,
) -> str:
    """
    Render a cortical surface map to PNG using nilearn's matplotlib backend.
    Views: left lateral, right lateral, dorsal.
    Returns the output path.
    """
    from tribev2.plotting.cortical import PlotBrainNilearn

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    plotter = PlotBrainNilearn(mesh="fsaverage5")

    if vmax is None:
        vmax = float(np.percentile(np.abs(preds_mean), 95))

    views = ["left", "right", "dorsal"]
    plotter.plot_surf(
        preds_mean,
        views=views,
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax if symmetric_cbar else 0,
        symmetric_cbar=symmetric_cbar,
        colorbar=True,
        norm_percentile=None,
    )
    fig = plt.gcf()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_delta_brain_png(
    preds_mean_a: np.ndarray,
    preds_mean_b: np.ndarray,
    output_path: str,
) -> str:
    """
    Render a B − A difference map. Red = B more active, blue = A more active.
    """
    delta = preds_mean_b - preds_mean_a
    return render_brain_png(
        delta,
        output_path=output_path,
        cmap="RdBu_r",
        symmetric_cbar=True,
    )


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from tribev2 import TribeModel

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python brain.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Loading model...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    print(f"Running inference on {video_path}...")
    preds, segments = run_inference(video_path, model)
    print(f"preds shape: {preds.shape}")
    stats = compute_stats(preds, segments)
    print(f"Overall score: {stats['overall_score']:.6f}")
    print(f"Peak at: {stats['peak_timestamp_s']:.1f}s")
    print(f"Top ROIs: {stats['top_rois'][:5]}")
    print(f"Visual score: {stats['visual_score']:.6f}")
    print(f"Language score: {stats['language_score']:.6f}")
    print(f"Attention score: {stats['attention_score']:.6f}")

    png_path = "outputs/renders/test_brain.png"
    render_brain_png(stats["preds_mean"], png_path)
    print(f"Brain PNG saved to: {png_path}")
