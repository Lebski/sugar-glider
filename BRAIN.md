# How brain.py works

`src/brain.py` sits between the TRIBE v2 model and the Streamlit UI. It handles inference, caching, statistics, and rendering — everything that turns a raw `.mp4` into the numbers and images shown in the app.

---

## Pipeline overview

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

---

## What TRIBE v2 actually predicts

TRIBE v2 was trained on fMRI recordings of people watching naturalistic videos. Given a new video it has never seen, it predicts what a **group-average brain** would look like in the scanner.

The output is a 2D array:

```
preds.shape == (n_segments, 81924)
               ^^^^^^^^^^^  ^^^^^^
               time         space
```

- **Rows** — one per TR (repetition time ≈ 2 seconds). A 30-second ad produces ~15 rows.
- **Columns** — one per vertex on the `fsaverage5` cortical surface mesh (40,962 per hemisphere × 2). Each value is a predicted **BOLD signal** — positive means more activation than baseline, negative means less.

> All scores displayed in the UI are multiplied by 1000 (milli-BOLD units) to keep the numbers readable. Raw values are typically in the range 0.001–0.01.

---

## Metrics

### Overall Brain Score

```python
preds.mean()   # scalar — mean BOLD across all vertices and all TRs
```

The headline number. Higher = more total cortical engagement across the whole ad. Used to determine the winner in a head-to-head comparison.

---

### Ad Impact Score

A composite score designed to predict purchase intent more directly than raw whole-brain BOLD. It averages activation across the brain regions most strongly linked to advertising effectiveness: the **attention network** and **cortical reward-adjacent areas**.

**Attention network** — top-down attentional control:
`FEF · 7PC · VIP · LIPv · LIPd · IPS1`

**Cortical reward-adjacent regions** — areas tightly coupled to the dopamine/reward circuit that lie on the cortical surface. (The Nucleus Accumbens and ventral striatum are subcortical structures and are not predicted by TRIBE v2's cortical mesh model.)

| Region group | HCP areas | Marketing relevance |
|---|---|---|
| Orbitofrontal Cortex | 47l · 13l · 11l · 47s | Value computation, willingness to pay |
| vmPFC / mPFC | 11m · 25 · 10v | Reward anticipation, self-relevance |
| Anterior Cingulate | p24 · a24 · d32 | Motivation, emotional salience |
| Insula | Ig · PoI1 · AVI · AAIC | Interoceptive salience, gut-feeling response |
| Temporal pole | TGd · TGv | Emotional memory, brand/social recognition |

```python
impact_score = _safe_roi_mean(preds_mean, IMPACT_ROIS)
```

An ad that scores high here is capturing sustained attention **and** activating the brain's value-assessment and emotional-memory systems — a stronger predictor of conversion than whole-brain engagement alone.

---

### Early Attention Score

Weights brain activation in the **first seconds of the ad more heavily** than later moments using logarithmic decay, based on the principle that viewer attention is front-loaded.

```
w_i = log(N+1) − log(i+1)       i = segment index (0 = first), N = total segments

early_attention_score = Σ(w_i × engagement_i) / Σ(w_i)
```

In a 30s ad (15 TRs): segment 0 (0–2s) carries weight ≈ 2.77, segment 14 (28–30s) carries weight ≈ 0.06 — roughly 45× less.

**How to read it:**
- `early_attention_score > overall_score` → the opening is outperforming the rest of the ad
- `early_attention_score < overall_score` → the ad builds and gets stronger as it goes

---

### Engagement over time

```python
preds.mean(axis=1)   # shape (n_segments,) — mean activation per TR
```

Mean BOLD collapsed across all cortical vertices, one value per ~2s segment. Plotted as the engagement timeline in the UI. Also drives the **real-time stats**: as the video plays, the UI shows the current segment's activation value relative to the ad's mean.

---

### Peak Moment

```python
timestamps[np.argmax(preds.mean(axis=1))]   # seconds
```

The timestamp of the segment with the highest whole-brain mean activation — the moment the ad drives the strongest neural response.

---

### Cognitive breakdown

Three grouped scores mapping brain systems to marketing questions:

| Score | Brain regions | Question it answers |
|---|---|---|
| **Visual cortex** | V1 · V2 · V3 · V4 · MT · MST · V3A · V3B | Is the visual creative landing? |
| **Language cortex** | 44 · 45 · STSdp · STSda · STSvp · STSva · TE1a · TE1m | Is the voiceover or copy being processed? |
| **Attention network** | FEF · 7PC · VIP · LIPv · LIPd · IPS1 | Is the viewer paying attention? |

Each score is the mean of `preds_mean` across all vertices in those regions.

---

### Per-segment top regions

For every TR in the ad, the top 3 activated HCP regions are precomputed:

```python
for i in range(n_segments):
    seg_roi = summarize_by_roi(preds[i])
    top3_idx = np.argsort(seg_roi)[::-1][:3]
    per_segment_top_rois.append([roi_names[j] for j in top3_idx])
```

Shown in the UI as "Active regions" and updated in real time as the video plays.

---

### Comparison (B − A)

```python
delta = preds_mean_b - preds_mean_a   # shape (81924,) — per-vertex difference
```

Computed for every vertex. Rendered as the **difference brain map** (red = B stronger, blue = A stronger). Also summarised per ROI for the "Where they differ most" bar chart.

Winner is determined by `overall_score`. Tie threshold: `|delta| < 1e-6`.

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
| MST | Medial Superior Temporal Area | Visual | Optic flow and self-motion perception; wide-field camera movement |
| DVT | Dorsal Visual Transition Area | Visual | Bridge between motion areas and parietal attention regions |
| **Language** ||||
| 44 | Broca's Area BA44 | Language | Core speech production; activated by voiceover and dialogue |
| 45 | Broca's Area BA45 | Language | Speech comprehension; active when processing meaning of spoken words |
| STSdp / STSda | Superior Temporal Sulcus — dorsal | Language | Integrates audio-visual speech; processes talking faces and lip sync |
| STSvp / STSva | Superior Temporal Sulcus — ventral | Language | Higher-level semantic integration of spoken language in context |
| TE1a / TE1m / TE1p | Temporal Area TE1 | Language | Auditory association cortex; voice identity, tone, non-speech sounds |
| **Attention** ||||
| FEF | Frontal Eye Field | Attention | Controls voluntary gaze and directs attention to salient screen regions |
| IPS1 | Intraparietal Sulcus Area 1 | Attention | Holds the attentional spotlight; tracks objects across time |
| VIP | Ventral Intraparietal Area | Attention | Integrates visual, tactile, auditory signals; responds to salient stimuli |
| LIPv / LIPd | Lateral Intraparietal Area | Attention | Encodes priority maps — where attention should go next |
| 7PC | Parietal Area 7PC | Attention | Top-down attentional control and working memory for visual locations |
| **Reward / Impact** ||||
| 47l / 13l / 11l / 47s | Orbitofrontal Cortex | Reward | Computes subjective value and expected reward; predicts willingness to pay |
| 11m / 25 / 10v | vmPFC / mPFC | Reward | Self-referential processing and reward anticipation; active when content feels personally relevant |
| p24 / a24 / d32 | Anterior Cingulate Cortex | Reward | Motivational salience and effort allocation; bridges emotion and action |
| Ig / PoI1 / AVI / AAIC | Insula | Reward | Interoceptive awareness and emotional salience — the neurological basis of gut feeling |
| TGd / TGv | Temporal Pole | Reward | Links perception to emotional memory; key for brand familiarity and social recognition |
| **Other** ||||
| TPOJ1 / TPOJ2 / TPOJ3 | Temporo-Parieto-Occipital Junction | Multisensory | Integrates audio-visual inputs; social cognition and perspective-taking |
| PH / PGp | Parieto-occipital areas | Spatial | Scene perception and spatial layout processing |

---

## Caching

Three layers keep repeated inference off the hot path:

| Layer | Location | Lifetime |
|---|---|---|
| Session state | `st.session_state` | Browser tab |
| Disk cache (stats) | `outputs/client_cache/<hash>/` | Permanent |
| Inference cache | `outputs/cache/<hash>.pkl` | Permanent |

The SHA256 hash is computed from the video file bytes, so the same video always hits cache regardless of filename or upload path.

Brain PNGs are saved to `outputs/renders/` and reused the same way.

If a cached result is missing a score that was added in a later version (e.g. `impact_score`), it is **backfilled on load** from the stored `preds_mean.npy` without re-running inference.
