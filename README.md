# sugar-glider

A marketing tool for comparing how video ads drive brain responses, built on [TRIBE v2](https://github.com/facebookresearch/tribev2) — Meta's foundation model for predicting fMRI brain activity from video, audio, and text.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **ffmpeg** — required by WhisperX for audio decoding:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- A HuggingFace account with access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- A GPU is strongly recommended — inference on CPU takes several minutes per video

---

## Installation

```bash
uv sync
```

Authenticate with HuggingFace — either via CLI:

```bash
huggingface-cli login
```

Or by adding your token to a `.env` file at the project root:

```
HF_TOKEN=hf_...
```

> **Note:** LLaMA 3.2-3B is a gated model. Request access at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) before running — approval is usually instant.

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

**Side-by-side brain maps** — cortical surface images showing mean predicted BOLD activation across the duration of each ad (left hemisphere, right hemisphere, dorsal view).

**Statistics panel** per ad:
- Brain Score, Ad Impact Score, Early Attention Score, Peak Moment
- Real-time segment stats — as the video plays, activation and top active regions update every ~2s
- Engagement timeline — activation per ~2s segment with peak and playback cursor
- Top 10 activated brain regions (HCP MMP1.0 parcellation)
- Cognitive breakdown: visual cortex · language cortex · attention network

**Comparison section**:
- Winner badge with score delta
- B − A difference brain map (red = B stronger, blue = A stronger)
- Top regions where the two ads diverge most
- Full region legend with abbreviations, full names, and descriptions

---

## Project structure

```
sugar-glider/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── api.py              # FastAPI REST API (optional)
│   ├── brain.py            # Inference, caching, stats, rendering
│   ├── library.py          # Ad library (scans ads/)
│   ├── result_cache.py     # Persistent disk cache for analysis results
│   └── components/
│       └── video_player/   # Custom Streamlit component for real-time sync
│           └── index.html
├── ads/                    # Your .mp4 ad files go here
├── cache/                  # TRIBE v2 model weights (auto-downloaded)
├── outputs/
│   ├── cache/              # Raw inference results cached by file hash (.pkl)
│   ├── client_cache/       # Processed stats cached by file hash
│   └── renders/            # Brain PNG files
└── uploads/                # Temp storage for drag-and-drop uploads
```

---

## How it works

### Pipeline

```
.mp4 file
   │
   ▼
get_events_dataframe()        extract audio, transcribe speech with WhisperX
   │                          → events DataFrame (video/audio/word timings)
   ▼
model.predict(events)         TRIBE v2 forward pass
   │                          → preds: (n_segments × 81,924 vertices)
   ▼
compute_stats(preds)          aggregate into marketing metrics
   │
   ├── render_brain_png()     → cortical surface PNG (mean activation)
   └── compute_delta_stats()  → comparison between two ads
```

### What TRIBE v2 predicts

TRIBE v2 was trained on fMRI recordings of people watching naturalistic videos. Given a new video, it predicts what a **group-average brain** would look like in the scanner.

The output is a 2D array:

```
preds.shape == (n_segments, 81924)
               ^^^^^^^^^^^  ^^^^^^
               time         space
```

- **Rows** — one per TR (repetition time ≈ 2 seconds). A 30-second ad produces ~15 rows.
- **Columns** — one per vertex on the `fsaverage5` cortical surface mesh (40,962 per hemisphere × 2). Each value is a predicted **BOLD signal** — positive means more activation than baseline, negative means less.

> All scores in the UI are multiplied by 1000 (milli-BOLD) to keep numbers readable. Raw values are typically in the range 0.001–0.01.

---

## Metrics

### Brain Score

```python
preds.mean()   # mean BOLD across all vertices and all TRs
```

The headline number. Higher = more total cortical engagement across the whole ad. Used to determine the winner in a head-to-head comparison.

---

### Ad Impact Score

A composite score designed to predict purchase intent more directly than raw whole-brain BOLD. Averages activation across the regions most strongly linked to advertising effectiveness: the **attention network** and **cortical reward-adjacent areas**.

**Attention network** — top-down attentional control:
`FEF · 7PC · VIP · LIPv · LIPd · IPS1`

**Cortical reward-adjacent regions** — areas tightly coupled to the dopamine/reward circuit. (The Nucleus Accumbens and ventral striatum are subcortical and not predicted by TRIBE v2's cortical mesh.)

| Region group | HCP areas | Marketing relevance |
|---|---|---|
| Orbitofrontal Cortex | 47l · 13l · 11l · 47s | Value computation, willingness to pay |
| vmPFC / mPFC | 11m · 25 · 10v | Reward anticipation, self-relevance |
| Anterior Cingulate | p24 · a24 · d32 | Motivation, emotional salience |
| Insula | Ig · PoI1 · AVI · AAIC | Interoceptive salience, gut-feeling response |
| Temporal pole | TGd · TGv | Emotional memory, brand/social recognition |

An ad that scores high here is capturing sustained attention **and** activating the brain's value-assessment and emotional-memory systems — a stronger predictor of conversion than whole-brain engagement alone.

---

### Early Attention Score

Weights activation in the **first seconds of the ad more heavily** using logarithmic decay — viewer attention and purchase intent are front-loaded.

```
w_i = log(N+1) − log(i+1)       i = segment index (0 = first), N = total segments

early_attention_score = Σ(w_i × engagement_i) / Σ(w_i)
```

In a 30s ad (15 TRs): segment 0 (0–2s) carries ~45× the weight of the final segment.

- `early_attention_score > brain_score` → the opening outperforms the rest
- `early_attention_score < brain_score` → the ad builds and gets stronger as it goes

---

### Engagement over time

```python
preds.mean(axis=1)   # shape (n_segments,) — mean activation per ~2s TR
```

Plotted as the engagement timeline. Drives the **real-time stats**: as the video plays, the UI shows the current segment's activation delta vs the ad mean, and the top 3 active brain regions for that moment.

---

### Peak Moment

The timestamp of the TR with the highest whole-brain mean activation — when the ad drives the strongest neural response.

---

### Cognitive breakdown

| Score | Brain regions | Question it answers |
|---|---|---|
| **Visual cortex** | V1 · V2 · V3 · V4 · MT · MST · V3A · V3B | Is the visual creative landing? |
| **Language cortex** | 44 · 45 · STSdp · STSda · STSvp · STSva · TE1a · TE1m | Is the voiceover or copy being processed? |
| **Attention network** | FEF · 7PC · VIP · LIPv · LIPd · IPS1 | Is the viewer paying attention? |

---

### Comparison (B − A)

```python
delta = preds_mean_b - preds_mean_a   # per-vertex difference across 81,924 vertices
```

Rendered as a difference brain map and summarised per ROI for the "Where they differ most" bar chart. Winner is whichever ad has the higher Brain Score.

---

## HCP region glossary

| Abbreviation | Full name | System | Description |
|---|---|---|---|
| **Visual** ||||
| V1 | Primary Visual Cortex | Visual | First cortical stage of vision — edges, contrast, basic orientation |
| V2 | Secondary Visual Cortex | Visual | Passes signals to both ventral (what) and dorsal (where) streams |
| V3 / V3A / V3B | Visual Area V3 | Visual | Motion and depth; V3A strongly driven by motion and optical flow |
| V4 | Visual Area V4 | Visual | Colour, shape, object form — critical for brand colour and logo processing |
| MT | Middle Temporal Area | Visual | Dedicated motion processing; responsive to fast cuts and moving objects |
| MST | Medial Superior Temporal Area | Visual | Optic flow and self-motion; responds to wide-field camera movement |
| DVT | Dorsal Visual Transition Area | Visual | Bridge between motion areas and parietal attention regions |
| **Language** ||||
| 44 | Broca's Area BA44 | Language | Core speech production; activated by voiceover and dialogue |
| 45 | Broca's Area BA45 | Language | Speech comprehension; processing the meaning of spoken words |
| STSdp / STSda | Superior Temporal Sulcus — dorsal | Language | Integrates audio-visual speech; processes talking faces and lip sync |
| STSvp / STSva | Superior Temporal Sulcus — ventral | Language | Higher-level semantic integration of spoken language in context |
| TE1a / TE1m / TE1p | Temporal Area TE1 | Language | Auditory association cortex; voice identity, tone, non-speech sounds |
| **Attention** ||||
| FEF | Frontal Eye Field | Attention | Controls voluntary gaze and directs attention to salient screen regions |
| IPS1 | Intraparietal Sulcus Area 1 | Attention | Holds the attentional spotlight; tracks objects across time |
| VIP | Ventral Intraparietal Area | Attention | Integrates visual, tactile, and auditory signals; salience detection |
| LIPv / LIPd | Lateral Intraparietal Area | Attention | Encodes priority maps — where attention should move next |
| 7PC | Parietal Area 7PC | Attention | Top-down attentional control and working memory for visual locations |
| **Reward / Impact** ||||
| 47l / 13l / 11l / 47s | Orbitofrontal Cortex | Reward | Computes subjective value and expected reward; predicts willingness to pay |
| 11m / 25 / 10v | vmPFC / mPFC | Reward | Self-referential processing and reward anticipation; active when content feels personally relevant |
| p24 / a24 / d32 | Anterior Cingulate Cortex | Reward | Motivational salience and effort allocation; bridges emotion and action |
| Ig / PoI1 / AVI / AAIC | Insula | Reward | Interoceptive awareness and emotional salience — the neurological basis of gut feeling |
| TGd / TGv | Temporal Pole | Reward | Links perception to emotional memory; key for brand familiarity and social recognition |
| **Other** ||||
| TPOJ1 / TPOJ2 | Temporo-Parieto-Occipital Junction | Multisensory | Integrates audio-visual inputs; social cognition and perspective-taking |
| PH / PGp | Parieto-occipital areas | Spatial | Scene perception and spatial layout processing |

---

## Caching

Three layers keep repeated inference off the hot path:

| Layer | Location | Lifetime |
|---|---|---|
| Session state | `st.session_state` | Browser tab |
| Disk cache (stats) | `outputs/client_cache/<hash>/` | Permanent |
| Inference cache | `outputs/cache/<hash>.pkl` | Permanent |

The SHA256 hash is computed from the video file bytes, so the same video always hits cache regardless of filename or upload path. If a cached result is missing a score added in a later version, it is backfilled on load from the stored `preds_mean.npy` — no re-inference needed.

---

## REST API (optional)

A FastAPI wrapper is available for programmatic access or remote inference.

```bash
uv run uvicorn src.api:app --reload
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Check if the model is loaded and ready |
| `POST` | `/analyze` | Upload a video file, returns stats + brain PNG (base64) |
| `POST` | `/predict` | Run inference on a server-side file path (legacy) |

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@campaign-a.mp4"
```
