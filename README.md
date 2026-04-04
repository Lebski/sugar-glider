# sugar-glider

A marketing tool for comparing how video ads drive brain responses, built on [TRIBE v2](https://github.com/facebookresearch/tribev2) — Meta's foundation model for predicting fMRI brain activity from video, audio, and text.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **ffmpeg** — required by WhisperX for audio decoding:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- A HuggingFace account with access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

## Installation

```bash
uv sync
```

Then authenticate with HuggingFace — either via CLI:

```bash
huggingface-cli login
```

Or by adding your token to a `.env` file at the project root:

```
HF_TOKEN=hf_...
```

> **Note:** LLaMA 3.2-3B is a gated model. Request access at
> [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
> before running — approval is usually instant.

---

## Usage

### 1. Add your ads

Drop `.mp4` files into the `ads/` folder at the project root:

```
ads/
  campaign-a.mp4
  campaign-b.mp4
```

### 2. Launch the app

```bash
uv run streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 3. Compare

Select two ads from the library (or upload them directly), then click **Compare**. Results are cached on disk — re-running the same video is instant.

---

## What you get

**Side-by-side brain maps** — cortical surface images showing mean predicted brain activation across the duration of each ad (left hemisphere, right hemisphere, dorsal view).

**Statistics panel** (toggle per ad):
- Overall brain engagement score
- Engagement timeline — activation per ~2s segment, with peak moment highlighted
- Top activated brain regions (HCP parcellation)
- Cognitive breakdown: visual cortex · language cortex · attention network

**Comparison section**:
- Winner badge
- B − A difference brain map (red = B stronger, blue = A stronger)
- Top regions where the two ads diverge most

---

## Project structure

```
sugar-glider/
├── src/
│   ├── app.py        # Streamlit UI
│   ├── api.py        # FastAPI REST API (optional)
│   ├── brain.py      # Inference, caching, stats, rendering
│   ├── library.py    # Ad library (scans ads/)
│   └── test.py       # API smoke test
├── ads/              # Your .mp4 ad files go here
├── examples/         # Demo video (earth.mp4)
├── cache/            # TRIBE v2 model weights (auto-downloaded)
├── outputs/
│   ├── cache/        # Inference results cached by file hash
│   └── renders/      # Brain PNG files
└── uploads/          # Temp storage for drag-and-drop uploads
```

---

## REST API (optional)

A FastAPI wrapper is available for programmatic access.

```bash
uv run uvicorn src.api:app --reload
```

| Method | Path       | Description                            |
| ------ | ---------- | -------------------------------------- |
| `GET`  | `/health`  | Check if the model is loaded and ready |
| `POST` | `/predict` | Predict brain responses to a video     |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"video_path": "examples/earth.mp4"}'
```

---

## Notes

- Model weights (~several GB) are downloaded automatically on first run into `cache/`
- Inference runs on CPU by default — expect a few minutes per video on a MacBook
- Predictions use the "average subject" on the fsaverage5 cortical mesh (~82k vertices)
