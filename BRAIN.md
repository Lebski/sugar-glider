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
   ├── render_brain_png()     → cortical surface PNG
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

---

## Metrics tracked

### Overall Brain Score
```python
preds.mean()   # scalar — mean BOLD activation across all vertices and all TRs
```
The single headline number. Higher = more total cortical engagement across the whole ad. Used to determine the winner in a comparison.

> Note: this is an absolute BOLD estimate, not normalised to 0–100. Raw values are typically small (e.g. 0.002), but what matters is which ad scores higher.

---

### Engagement over time
```python
preds.mean(axis=1)   # shape (n_segments,) — mean activation per TR
```
Mean activation collapsed across all brain vertices, one value per ~2s segment. Plotted as the engagement timeline in the UI. The segment with the highest value is the **Peak Moment**.

---

### Early Attention Score (log-weighted)

A custom metric that weights brain activation in the **first seconds of the ad more heavily** than later moments, using logarithmic decay:

```
w_i = log(N+1) − log(i+1)       i = segment index (0 = first), N = total segments

early_attention_score = Σ(w_i × engagement_i) / Σ(w_i)
```

In a 30s ad (15 TRs): segment 0 (0–2s) carries weight ≈ 2.77, segment 14 (28–30s) carries weight ≈ 0.06 — roughly 45× less.

```python
weights = np.array([np.log(N + 1) - np.log(i + 1) for i in range(N)])
early_attention_score = float(np.dot(weights, engagement_over_time) / weights.sum())
```

**Why this matters:** viewer attention and purchase intent are heavily front-loaded. An ad that spikes neural engagement at the start beats one that builds slowly, even if their `overall_score` is similar. If `early_attention_score > overall_score`, the opening is outperforming the rest. If it's lower, the ad gets better as it goes.

---

### Peak Moment
```python
timestamps[np.argmax(preds.mean(axis=1))]   # seconds into the ad
```
The timestamp of the TR with the highest whole-brain mean activation. Useful for identifying which part of the ad drives the strongest neural response.

---

### Brain regions (HCP parcellation)

The 81,924 vertices are grouped into **180 named cortical areas** using the [HCP MMP1.0 parcellation](https://www.nature.com/articles/nature18933) (Glasser et al., 2016). Each area covers a functionally distinct patch of cortex.

Per-ROI score:
```python
summarize_by_roi(preds_mean)   # mean of preds_mean within each of the 180 ROIs
```

Top activated regions:
```python
get_topk_rois(preds_mean, k=10)   # names of the 10 highest-scoring ROIs
```

---

### Cognitive breakdown scores

Three grouped scores that map brain regions to things a marketer cares about:

| Score | Brain regions | What it means |
|---|---|---|
| **Visual cortex** | V1, V2, V3, V4, MT, MST, V3A, V3B | How much the visual creative is being processed — motion, colour, objects |
| **Language cortex** | 44 (Broca), 45 (Broca), STSdp, STSda, STSvp, STSva, TE1a, TE1m | How much the spoken or written message is being processed |
| **Attention network** | FEF, 7PC, VIP, LIPv, LIPd, IPS1 | How much the brain's top-down attention system is engaged |

Each score is the mean of `preds_mean` across all vertices belonging to those ROIs. Higher = more activation in that cognitive system.

---

### Comparison (B − A)

```python
delta = preds_mean_b - preds_mean_a   # shape (81924,) — per-vertex difference
```

Computed for every vertex on the surface. Rendered as the **difference brain map** (red = B stronger, blue = A stronger). Also summarised per ROI to produce the **Top differentiating regions** bar chart.

Winner is simply whichever ad has a higher `overall_score`.

---

## Caching

Inference is slow on CPU (several minutes per video). Results are saved to `outputs/cache/<sha256>.pkl` after the first run. The SHA256 is computed from the file bytes, so the same video always hits the cache regardless of filename.

Brain PNGs are saved to `outputs/renders/` and reused the same way.
