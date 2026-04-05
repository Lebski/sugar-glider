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
| **Visual Cortex** ||||
| V1 | Primary Visual Cortex | Visual | First cortical stage of vision. Neurons tuned for orientation, edges, contrast, spatial frequency, and binocular disparity. Also processes basic motion direction. |
| V2 | Secondary Visual Cortex | Visual | Organized into functional stripes that independently process colour, stereoscopic depth, and form. Computes illusory contours and figure-ground segregation. Distributes signals to both ventral ("what") and dorsal ("where") streams. |
| V3 | Visual Area V3 | Visual | Heterogeneous area processing orientation, colour, disparity, and motion. Distinct from V3A and V3B — the HCP atlas treats all three as separate areas with independent retinotopic maps. |
| V3A | Visual Area V3A | Visual | Strongly driven by motion and optical flow. Responds to coherent motion patterns and depth-from-motion cues. More motion-selective than V3 proper. |
| V3B | Visual Area V3B | Visual | Linked to stereoscopic depth and 3D structure-from-motion. Emphasises binocular disparity processing over motion. |
| V4 | Visual Area V4 | Visual | Key area for colour and intermediate-complexity shape. Contains "glob" (colour-selective) and "interglob" (shape-selective) subregions. First area where robust attentional modulation was demonstrated. Brand/logo recognition requires higher inferotemporal areas beyond V4. |
| MT | Middle Temporal Area (V5) | Visual | The cortex's primary motion-processing hub — direction-selective neurons predominate. Microstimulation alters perceived motion direction; lesions cause motion blindness. Responds to fast cuts and motion onsets. |
| MST | Medial Superior Temporal Area | Visual | Dorsal subdivision (MSTd) contains neurons tuned to expanding, contracting, and rotating flow fields. Integrates visual self-motion with vestibular signals. Wide-field camera movement maps directly to this function. |
| DVT | Dorsal Transitional Visual Area | Visual / Spatial | A newly described HCP area primarily part of the retrosplenial scene network, involved in spatial scene processing and navigation. Connects to dorsal-stream motion areas and parietal regions; core role is scene-spatial processing, not simply a motion-to-attention relay. |
| **Language** ||||
| 44 / 45 | Broca's Area (pars opercularis / pars triangularis) | Language | BA44 is linked to phonological processing, syntactic structure building, and articulatory planning. BA45 to controlled semantic retrieval. The production/comprehension split is an outdated simplification — both contribute to both. Also involved in working memory and cognitive control. |
| STSdp / STSda | Superior Temporal Sulcus — dorsal | Language / Audiovisual | Most consistently identified region for audiovisual speech integration. Bimodal neurons respond to both auditory and visual speech. Processes talking faces, lip sync, and voice-face binding. |
| STSvp / STSva | Superior Temporal Sulcus — ventral | Language / Multimodal Semantic | Receives strong visual inputs from inferior temporal cortex and parietal areas. Functions as a multimodal semantic integration zone — binding visual object/scene information with language meaning. |
| TE1a / TE1m | Temporal Area TE1 | Multimodal / Ventral Visual Stream | Located on the middle temporal gyrus. Primary function is high-level visual object and face processing at the end of the ventral "what" stream — not auditory association cortex. Connect to language regions via the arcuate fasciculus; voice identity belongs to temporal voice areas along the STS. |
| **Attention** ||||
| FEF | Frontal Eye Field | Attention (Dorsal Network) | Controls voluntary saccadic eye movements. Sends top-down feedback to visual cortex (especially V4) enhancing processing at attended locations. Part of the dorsal frontoparietal attention network. Salience-driven (bottom-up) attention is more the province of the ventral attention network (TPJ). |
| IPS1 | Intraparietal Sulcus Area 1 | Attention | Responds poorly to unattended stimuli. Linked to sustained spatial attention and visual short-term memory. |
| VIP | Ventral Intraparietal Area | Attention / Multisensory | Neurons respond to visual, tactile, and auditory stimulation with receptive fields anchored to peripersonal space. Also integrates vestibular signals for heading discrimination. |
| LIPd | Lateral Intraparietal Area — dorsal | Attention / Oculomotor | Integrates bottom-up salience and top-down biases to determine where gaze should go (priority map). LIPd primarily encodes saccade intentions. |
| LIPv | Lateral Intraparietal Area — ventral | Attention | Contributes to both attentional and oculomotor processes alongside LIPd. |
| 7PC | Parietal Area 7PC | Attention / Visuomotor | Visuomotor integration hub receiving inputs from intraparietal areas and somatosensory cortex, with projections to premotor regions. Part of the dorsal attention network; primary top-down control comes from DLPFC and FEF. |
| **Reward / Impact** ||||
| 13l / 11l | Medial Orbitofrontal Cortex | Reward / Value | Where economic value computation actually occurs. Neurons encode subjective value; activity correlates with willingness to pay. Represents hedonic value across modalities — food, money, social, aesthetic. |
| 47l / 47s | Lateral Orbitofrontal Cortex | Language / Cognitive Control | Distinct from medial OFC value regions. Involved in non-reward expectation violations, behavioural inhibition, and language-related semantic processing — not the same functional subsystem as 13l/11l. |
| 10v | Area 10v (vmPFC) | Reward / Self-Referential | Linked to self-referential processing, outcome evaluation, and expected value computation. Active when content feels personally relevant. Reward anticipation per se is more strongly associated with the ventral striatum; vmPFC computes expected value. |
| 25 | Subgenual Cingulate Cortex | Emotion / Autonomic | Critical node for mood regulation — hyperactivity is a biomarker for treatment-resistant depression. Involved in autonomic regulation and negative affect, not reward anticipation. |
| p24 / a24 / d32 | Pregenual Anterior Cingulate Cortex | Reward Evaluation / Emotion | Pregenual ACC areas associated with reward evaluation, emotional processing, and interoception. Effort allocation implicates dorsal ACC (24dd/24dv), not these pregenual parcels. |
| Ig / PoI1 / AVI / AAIC | Insula (posterior → anterior) | Interoception / Salience | Posterior insula (Ig, PoI1) receives primary interoceptive and somatosensory input. Anterior insula (AVI, AAIC) supports conscious interoceptive awareness and emotional salience; core node of the salience network. Key gradient: posterior = raw body signals; anterior = conscious awareness. |
| TGd / TGv | Temporal Pole (dorsal / ventral) | Social / Emotional Memory | Binds complex perceptual inputs to visceral emotional responses. Involved in face-to-person memory linking and theory of mind. Left hemisphere more verbal-semantic; right more face/emotion. |
| **Other** ||||
| TPOJ1 / TPOJ2 | Temporo-Parieto-Occipital Junction | Multisensory / Social | TPOJ1 links higher auditory and higher visual areas. The broader TPJ is well-established for social cognition, theory of mind, perspective-taking, and detecting expectation violations. |
| PGp | Inferior Parietal Cortex (angular gyrus region) | Spatial / Scene | Classified under Inferior Parietal Cortex in the HCP atlas (not parieto-occipital). Strong selective activation for scenes. Involved in scene perception, spatial layout, and episodic memory retrieval. |
| PH | MT+ Complex — Lateral Occipitotemporal Area | Visual / Object | Part of the MT+ complex in the HCP atlas (not parieto-occipital). Activated by tools and body parts. Not to be confused with PHA1–3 (parahippocampal areas), which are the actual scene-selective regions. |

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
