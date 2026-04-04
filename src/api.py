import base64
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is on the path when uvicorn is launched from the project root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import brain  # noqa: E402
from tribev2 import TribeModel  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

model: TribeModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global model
    if not os.environ.get("HF_TOKEN") and (
        token := os.environ.get("HUGGINGFACE_TOKEN")
    ):
        os.environ["HF_TOKEN"] = token
    logger.info("Loading TRIBE v2 model...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder="./cache"
    )
    logger.info("Model loaded and ready.")
    yield
    model = None


app = FastAPI(title="sugar-glider", lifespan=lifespan)


def _encode_array(arr: np.ndarray) -> str:
    """Encode a numpy float32 array as base64 for JSON transport."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _encode_png(path: str) -> str:
    """Read a PNG file and return it as base64."""
    return base64.b64encode(Path(path).read_bytes()).decode()


@app.get("/health")
def health():
    return {"status": "ready" if model is not None else "loading"}


API_CACHE_DIR = Path("outputs/api_cache")


def _api_cache_path(video_hash: str) -> Path:
    return API_CACHE_DIR / f"{video_hash}.json"


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept a video file upload, run TRIBE v2 inference, return
    stats + brain PNG + preds_mean (for client-side delta rendering).

    Responses are cached by video file hash — repeated uploads of the
    same file return instantly without re-running inference or rendering.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Read file into memory so we can hash it before writing to disk
    data = await file.read()
    video_hash = brain.hash_bytes(data)

    cache_path = _api_cache_path(video_hash)
    if cache_path.exists():
        logger.info("Cache hit for %s", video_hash[:12])
        import json
        return json.loads(cache_path.read_text())

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        preds, segments = brain.run_inference(tmp_path, model)
        stats = brain.compute_stats(preds, segments)

        png_path = f"outputs/renders/{video_hash}_mean.png"
        brain.render_brain_png(stats["preds_mean"], png_path)

        response = {
            "overall_score": stats["overall_score"],
            "peak_segment_idx": stats["peak_segment_idx"],
            "peak_timestamp_s": stats["peak_timestamp_s"],
            "engagement_over_time": stats["engagement_over_time"].tolist(),
            "segment_timestamps": stats["segment_timestamps"],
            "top_rois": stats["top_rois"],
            "roi_scores": stats["roi_scores"],
            "visual_score": stats["visual_score"],
            "language_score": stats["language_score"],
            "attention_score": stats["attention_score"],
            "brain_png_b64": _encode_png(png_path),
            "preds_mean_b64": _encode_array(stats["preds_mean"]),
        }

        API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import json
        cache_path.write_text(json.dumps(response))

        return response
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Legacy endpoint
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    video_path: str | None = None
    audio_path: str | None = None
    text_path: str | None = None


@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not any([request.video_path, request.audio_path, request.text_path]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: video_path, audio_path, text_path",
        )

    df = model.get_events_dataframe(
        video_path=request.video_path,
        audio_path=request.audio_path,
        text_path=request.text_path,
    )
    preds, segments = model.predict(events=df)

    return {
        "shape": list(preds.shape),
        "n_segments": len(segments),
        "predictions": preds.tolist(),
    }
