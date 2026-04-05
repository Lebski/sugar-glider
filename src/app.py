"""
Ad Brain Analyzer — Streamlit UI.

Two tabs:
  📚 Library — all known videos (ads/ + uploads). Upload, analyze, annotate.
  ⚖️  Compare — pick two analyzed ads and see the full brain comparison.

Run with: uv run streamlit run src/app.py
"""

import functools
import hashlib
import http.server
import os
import threading
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import brain
import history
import library
import result_cache

# HCP MMP1.0 region name glossary (abbreviation → full name)
HCP_ROI_NAMES = {
    "V1": "Primary Visual Cortex",
    "V2": "Secondary Visual Cortex",
    "V3": "Visual Area V3",
    "V4": "Visual Area V4",
    "V3A": "Visual Area V3A (dorsal stream)",
    "V3B": "Visual Area V3B",
    "MT": "Middle Temporal Area (motion perception)",
    "MST": "Medial Superior Temporal Area",
    "DVT": "Dorsal Visual Transition Area",
    "PIT": "Posterior Inferotemporal Area",
    "FST": "Fundus Superior Temporal Area",
    "LO1": "Lateral Occipital Area 1",
    "LO2": "Lateral Occipital Area 2",
    "LO3": "Lateral Occipital Area 3",
    "44": "Broca's Area BA44 (speech production)",
    "45": "Broca's Area BA45 (speech comprehension)",
    "STSdp": "Superior Temporal Sulcus — dorsal posterior",
    "STSda": "Superior Temporal Sulcus — dorsal anterior",
    "STSvp": "Superior Temporal Sulcus — ventral posterior",
    "STSva": "Superior Temporal Sulcus — ventral anterior",
    "TE1a": "Temporal Area TE1a (auditory association)",
    "TE1m": "Temporal Area TE1m (auditory association)",
    "TE1p": "Temporal Area TE1p (auditory association)",
    "FEF": "Frontal Eye Field (top-down visual attention)",
    "IPS1": "Intraparietal Sulcus Area 1",
    "VIP": "Ventral Intraparietal Area",
    "LIPv": "Lateral Intraparietal Area — ventral",
    "LIPd": "Lateral Intraparietal Area — dorsal",
    "7PC": "Parietal Area 7PC (attention)",
    "TPOJ1": "Temporo-Parieto-Occipital Junction 1",
    "TPOJ2": "Temporo-Parieto-Occipital Junction 2",
    "TPOJ3": "Temporo-Parieto-Occipital Junction 3",
    "PH": "Parieto-occipital Area PH",
    "PGp": "Parietal Area PGp",
    "IP1": "Intraparietal Area 1",
    "IP2": "Intraparietal Area 2",
}

load_dotenv()

st.set_page_config(
    page_title="Ad Brain Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

    div[data-testid="stMetric"] { background: #f9fafb; border-radius: 8px; padding: 0.75rem 1rem; }

    /* Unanalyzed card placeholder */
    .unanalyzed-thumb {
        width: 100%;
        aspect-ratio: 16/9;
        background: #e5e7eb;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #9ca3af;
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# File server (serves project root so the video component iframe can load videos)
# -----------------------------------------------------------------------

FILE_SERVER_PORT = 8765


@st.cache_resource(show_spinner=False)
def start_file_server() -> int:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(Path(".").resolve()),
    )
    try:
        server = http.server.HTTPServer(("127.0.0.1", FILE_SERVER_PORT), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except OSError:
        # Port already bound (e.g. after a hot-reload) — existing server still works
        pass
    return FILE_SERVER_PORT


_video_player_component = components.declare_component(
    "video_player",
    path=str(Path(__file__).parent / "components" / "video_player"),
)


def video_player(video_path: str, segment_timestamps: list) -> int:
    """Renders a synced video player. Returns the current segment index."""
    port = start_file_server()
    try:
        rel = Path(video_path).resolve().relative_to(Path(".").resolve())
        url = f"http://localhost:{port}/{rel.as_posix()}"
    except ValueError:
        st.video(video_path)
        return 0
    seg = _video_player_component(
        video_url=url, segment_timestamps=segment_timestamps, default=0
    )
    return seg if seg is not None else 0


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model():
    from tribev2 import TribeModel

    if not os.environ.get("HF_TOKEN") and (
        token := os.environ.get("HUGGINGFACE_TOKEN")
    ):
        os.environ["HF_TOKEN"] = token
    return TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def save_upload(uploaded_file) -> tuple[str, str] | tuple[None, None]:
    """Save an uploaded file to disk. Returns (path, hash)."""
    if uploaded_file is None:
        return None, None
    data = uploaded_file.read()
    h = hashlib.sha256(data).hexdigest()
    dest = Path("uploads") / f"{h}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return str(dest), h


def _backfill_scores(stats: dict) -> None:
    """Recompute scores added after a result was originally cached, in-place."""
    preds_mean = stats.get("preds_mean")
    if preds_mean is None:
        return
    if stats.get("impact_score") is None:
        stats["impact_score"] = brain._safe_roi_mean(preds_mean, brain.IMPACT_ROIS)
    if stats.get("early_attention_score") is None:
        eng = stats["engagement_over_time"]
        N = len(eng)
        if N > 1:
            weights = np.array([np.log(N + 1) - np.log(i + 1) for i in range(N)])
            stats["early_attention_score"] = float(np.dot(weights, eng) / weights.sum())
        else:
            stats["early_attention_score"] = float(eng[0]) if N == 1 else 0.0


def _register_history(video_hash: str, video_path: str, stats: dict) -> None:
    """Add/update this video in the persistent history index (once per session)."""
    sess_key = f"hist_registered_{video_hash}"
    if sess_key not in st.session_state:
        name = Path(video_path).stem
        history.add_or_update(video_hash, name, video_path, stats)
        st.session_state[sess_key] = True


def load_result(video_hash: str, video_path: str) -> dict | None:
    """Load an analysis result from disk cache — no model needed."""
    cached = result_cache.load(video_hash)
    if cached is None:
        return None
    cached["video_path"] = video_path
    cached["video_hash"] = video_hash
    _backfill_scores(cached["stats"])
    return cached


def analyze_video(video_path: str, video_hash: str, model) -> dict:
    """
    Run analysis on a video (or load from disk cache if already done).
    Registers the result in history. Used by Library tab.
    """
    if result_cache.exists(video_hash):
        result = load_result(video_hash, video_path)
        _register_history(video_hash, video_path, result["stats"])
        return result

    with st.status(f"Analyzing {Path(video_path).stem}…", expanded=True) as status:
        st.write("Extracting features from video…")
        preds, segments = brain.run_inference(video_path, model)
        st.write("Computing brain statistics…")
        stats = brain.compute_stats(preds, segments)
        st.write("Rendering brain map…")
        png_path = f"outputs/renders/{video_hash}_mean.png"
        brain.render_brain_png(stats["preds_mean"], png_path)
        st.write("Saving to cache…")
        result_cache.save(video_hash, stats, Path(png_path).read_bytes())
        status.update(label="Done.", state="complete")

    result = {
        "video_path": video_path,
        "video_hash": video_hash,
        "stats": stats,
        "brain_png": str(result_cache._dir(video_hash) / "brain.png"),
    }
    _register_history(video_hash, video_path, stats)
    return result


def _build_unified_library() -> list[dict]:
    """
    Return all known videos in display order:
      1. Library ads (from ads/) — analyzed ones first, then unanalyzed
      2. Uploaded ads (from history, not in ads/)

    Each entry:
      {name, path, source, history_entry (or None), is_analyzed}
    """
    hist_index = history.load_index()
    hist_by_resolved = {
        str(Path(e["video_path"]).resolve()): e for e in hist_index
    }

    lib_entries = library.get_library_entries()
    lib_resolved_paths: set[str] = set()
    lib_items: list[dict] = []

    for lib in lib_entries:
        resolved = str(Path(lib["path"]).resolve())
        lib_resolved_paths.add(resolved)
        hist_entry = hist_by_resolved.get(resolved)
        lib_items.append({
            "name": hist_entry["name"] if hist_entry else lib["name"],
            "path": lib["path"],
            "source": "library",
            "history_entry": hist_entry,
            "is_analyzed": hist_entry is not None,
        })

    # Sort: analyzed first, then by name
    lib_items.sort(key=lambda x: (not x["is_analyzed"], x["name"].lower()))

    upload_items: list[dict] = []
    for entry in hist_index:
        resolved = str(Path(entry["video_path"]).resolve())
        if resolved not in lib_resolved_paths:
            upload_items.append({
                "name": entry["name"],
                "path": entry["video_path"],
                "source": "upload",
                "history_entry": entry,
                "is_analyzed": True,
            })

    return lib_items + upload_items


# -----------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------

def engagement_chart(stats: dict, color: str, cursor_idx: int | None = None, yrange: tuple | None = None) -> go.Figure:
    times = stats["segment_timestamps"]
    values = stats["engagement_over_time"].tolist()
    peak_idx = stats["peak_segment_idx"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace("rgb(", "rgba(").replace(")", ", 0.08)"),
        marker=dict(size=5),
    ))
    if 0 <= peak_idx < len(times):
        fig.add_vline(
            x=times[peak_idx],
            line=dict(color=color, dash="dash", width=1),
            annotation_text=f"Peak {times[peak_idx]:.0f}s",
            annotation_position="top right",
            annotation_font_size=11,
        )
    if cursor_idx is not None and 0 <= cursor_idx < len(times):
        fig.add_vline(
            x=times[cursor_idx],
            line=dict(color="#6b7280", dash="dot", width=1.5),
            annotation_text=f"▶ {times[cursor_idx]:.0f}s",
            annotation_position="top left",
            annotation_font_size=10,
        )
    fig.update_layout(
        height=180, margin=dict(t=10, b=30, l=50, r=10),
        xaxis_title="Time (s)", yaxis_title="Activation",
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    yaxis_kwargs = dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    if yrange is not None:
        yaxis_kwargs["range"] = list(yrange)
    fig.update_yaxes(**yaxis_kwargs)
    return fig


def roi_bar_chart(stats: dict, color: str, top_n: int = 8) -> go.Figure:
    rois = stats["top_rois"][:top_n]
    scores = [stats["roi_scores"].get(r, 0) for r in rois]
    fig = go.Figure(go.Bar(
        x=scores, y=rois, orientation="h",
        marker_color=color, marker_line_width=0, marker_opacity=0.8,
    ))
    fig.update_layout(
        height=260, margin=dict(t=10, b=30, l=120, r=10),
        xaxis_title="Mean activation", yaxis=dict(autorange="reversed"),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def delta_roi_chart(delta_stats: dict) -> go.Figure:
    rois = delta_stats["top_differentiating_rois"]
    deltas = [delta_stats["roi_deltas"][r] for r in rois]
    colors = ["#ef4444" if d > 0 else "#3b82f6" for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=rois, orientation="h",
        marker_color=colors, marker_line_width=0, marker_opacity=0.8,
    ))
    fig.add_vline(x=0, line=dict(color="#9ca3af", width=1))
    fig.update_layout(
        height=260, margin=dict(t=10, b=30, l=120, r=10),
        xaxis_title="B − A  (red = B stronger · blue = A stronger)",
        yaxis=dict(autorange="reversed"),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# -----------------------------------------------------------------------
# Results panel
# -----------------------------------------------------------------------

def results_panel(label: str, result: dict, color: str, yrange: tuple | None = None):
    stats = result["stats"]
    seg_idx = 0

    if result.get("video_path") and Path(result["video_path"]).exists():
        seg_idx = video_player(result["video_path"], stats["segment_timestamps"])

    brain_tab, stats_tab = st.tabs(["Brain Map", "Statistics"])

    with brain_tab:
        if Path(result["brain_png"]).exists():
            st.image(result["brain_png"], use_container_width=True)
        else:
            st.caption("Brain map not available.")

    with stats_tab:
        seg_ts = stats["segment_timestamps"]
        seg_act = stats["engagement_over_time"]
        mean_act = float(seg_act.mean())
        current_val = float(seg_act[seg_idx]) if seg_idx < len(seg_act) else mean_act
        per_seg_rois = stats.get("per_segment_top_rois", [])
        top_rois_now = per_seg_rois[seg_idx] if seg_idx < len(per_seg_rois) else []
        ts_label = f"{seg_ts[seg_idx]:.0f}s" if seg_idx < len(seg_ts) else "0s"
        early = stats.get("early_attention_score")
        roi_help = "\n".join(
            f"{r}: {HCP_ROI_NAMES.get(r, 'HCP cortical area')}" for r in top_rois_now
        ) if top_rois_now else None

        st.metric("Brain Score (×10⁻³)", f"{stats['overall_score'] * 1000:.2f}", help="Mean predicted BOLD activation across all vertices and segments")
        impact = stats.get("impact_score")
        st.metric("Ad Impact Score (×10⁻³)", f"{impact * 1000:.2f}" if impact is not None else "—", help="Attention network + cortical reward-adjacent regions (OFC, vmPFC, ACC, Insula, Temporal pole)")
        st.metric("Early Attention (×10⁻³)", f"{early * 1000:.2f}" if early is not None else "—", help="Log-weighted score — first seconds count more")
        st.metric("Peak Moment", f"{stats['peak_timestamp_s']:.0f}s")
        st.metric("Duration", f"{len(seg_act)} TRs")
        st.divider()
        st.metric(f"Now · {ts_label} (×10⁻³)", f"{current_val * 1000:.2f}", delta=f"{(current_val - mean_act) * 1000:+.2f} vs mean")
        st.metric("Active regions", " · ".join(top_rois_now) if top_rois_now else "—", help=roi_help)

        st.markdown("**Engagement over time**")
        st.plotly_chart(
            engagement_chart(stats, color, cursor_idx=seg_idx, yrange=yrange),
            use_container_width=True, key=f"eng_{label}",
        )

        st.markdown("**Top activated regions**")
        st.plotly_chart(roi_bar_chart(stats, color), use_container_width=True, key=f"roi_{label}")

        with st.expander("Cognitive breakdown"):
            st.metric("Visual cortex (×10⁻³)", f"{stats['visual_score'] * 1000:.2f}", help="V1 · V2 · V3 · V4 · MT · MST · V3A · V3B")
            st.metric("Language cortex (×10⁻³)", f"{stats['language_score'] * 1000:.2f}", help="Broca (BA44/45) · STS · TE1a · TE1m")
            st.metric("Attention network (×10⁻³)", f"{stats['attention_score'] * 1000:.2f}", help="FEF · IPS1 · VIP · LIPv · LIPd · 7PC")


# -----------------------------------------------------------------------
# Region legend
# -----------------------------------------------------------------------

def region_legend():
    with st.expander("Region legend"):
        st.markdown("""
| Abbreviation | Full name | System | Description |
|---|---|---|---|
| **Visual cortex** ||||
| V1 | Primary Visual Cortex | Visual | First cortical stage of vision — edges, contrast, basic orientation |
| V2 | Secondary Visual Cortex | Visual | Passes visual signals to both ventral (what) and dorsal (where) streams |
| V3 / V3A / V3B | Visual Area V3 | Visual | Processes motion and depth; V3A is strongly driven by motion and optical flow |
| V4 | Visual Area V4 | Visual | Colour, shape, and object form — critical for brand colour and logo processing |
| MT | Middle Temporal Area | Visual | Dedicated motion processing; highly responsive to fast cuts and moving objects |
| MST | Medial Superior Temporal Area | Visual | Optic flow and self-motion perception; responds to wide-field camera movement |
| DVT | Dorsal Visual Transition Area | Visual | Bridge between motion areas and parietal attention regions |
| **Language** ||||
| 44 / 45 | Broca's Area | Language | Core speech production (44) and comprehension (45); activated by voiceover and dialogue |
| STSdp / STSda | Superior Temporal Sulcus — dorsal | Language | Integrates audio and visual speech cues; processes talking faces and lip sync |
| STSvp / STSva | Superior Temporal Sulcus — ventral | Language | Higher-level semantic integration of spoken language in context |
| TE1a / TE1m | Temporal Area TE1 | Language | Auditory association cortex; processes voice identity, tone, and non-speech sounds |
| **Attention** ||||
| FEF | Frontal Eye Field | Attention | Controls voluntary gaze and directs attention to salient regions of the screen |
| IPS1 | Intraparietal Sulcus Area 1 | Attention | Holds the attentional spotlight; tracks multiple objects across time |
| VIP | Ventral Intraparietal Area | Attention | Integrates visual, tactile, and auditory signals; responds to stimuli near the body |
| LIPv / LIPd | Lateral Intraparietal Area | Attention | Encodes priority maps — where in the scene attention should go next |
| 7PC | Parietal Area 7PC | Attention | Top-down attentional control and working memory for visual locations |
| **Reward / Impact** ||||
| 47l / 13l / 11l / 47s | Orbitofrontal Cortex | Reward | Computes subjective value and expected reward; predicts willingness to pay |
| 11m / 25 / 10v | vmPFC / mPFC | Reward | Self-referential processing and reward anticipation; active when content feels personally relevant |
| p24 / a24 / d32 | Anterior Cingulate Cortex | Reward | Signals motivational salience and effort allocation; bridges emotion and action |
| Ig / PoI1 / AVI / AAIC | Insula | Reward | Interoceptive awareness and emotional salience — the neurological basis of gut feeling |
| TGd / TGv | Temporal Pole | Reward | Links perception to emotional memory; key for brand familiarity and social recognition |
| **Other** ||||
| TPOJ1 / TPOJ2 | Temporo-Parieto-Occipital Junction | Multisensory | Integrates audio-visual inputs; involved in social cognition and perspective-taking |
| PH / PGp | Parieto-occipital areas | Spatial | Scene perception and spatial layout processing |
""")


# -----------------------------------------------------------------------
# Comparison section
# -----------------------------------------------------------------------

def comparison_section(result_a: dict, result_b: dict):
    stats_a = result_a["stats"]
    stats_b = result_b["stats"]
    delta_stats = brain.compute_delta_stats(stats_a, stats_b)
    winner = delta_stats["winner"]
    delta = delta_stats["score_delta"]

    st.divider()
    st.subheader("Comparison")

    if winner == "tie":
        st.info("Both ads produce nearly identical brain responses.")
    elif winner == "B":
        st.success(f"**Ad B wins** — {abs(delta):.5f} higher mean neural activation than Ad A.")
    else:
        st.success(f"**Ad A wins** — {abs(delta):.5f} higher mean neural activation than Ad B.")

    col_chart, col_map = st.columns([1, 2])

    with col_chart:
        st.markdown("**Where they differ most**")
        st.plotly_chart(delta_roi_chart(delta_stats), use_container_width=True, key="delta_chart")

    with col_map:
        st.markdown("**Brain difference map (B − A)**")
        ha = result_a["video_hash"]
        hb = result_b["video_hash"]
        delta_png = f"outputs/renders/{ha}_{hb}_delta.png"
        if not Path(delta_png).exists():
            with st.spinner("Rendering difference map…"):
                brain.render_delta_brain_png(stats_a["preds_mean"], stats_b["preds_mean"], delta_png)
        if Path(delta_png).exists():
            st.image(delta_png, use_container_width=True)
            st.caption("Red = Ad B more active · Blue = Ad A more active")


# -----------------------------------------------------------------------
# Library tab callbacks
# -----------------------------------------------------------------------

def _lib_name_cb(video_hash: str, key: str) -> None:
    history.update_meta(video_hash, name=st.session_state[key])


def _lib_desc_cb(video_hash: str, key: str) -> None:
    history.update_meta(video_hash, description=st.session_state[key])


# -----------------------------------------------------------------------
# Library tab
# -----------------------------------------------------------------------

def library_tab_view() -> None:
    # Upload widget — auto-analyzes on drop
    uploaded = st.file_uploader(
        "Upload a new ad (.mp4)",
        type=["mp4"],
        key="lib_uploader",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        video_path, video_hash = save_upload(uploaded)
        if video_path and video_hash:
            if result_cache.exists(video_hash):
                result = load_result(video_hash, video_path)
                _register_history(video_hash, video_path, result["stats"])
                st.success(f"**{Path(video_path).stem}** is already in the library.", icon="✓")
            else:
                with st.spinner("Loading model…"):
                    model = get_model()
                analyze_video(video_path, video_hash, model)
                st.rerun()

    st.divider()

    items = _build_unified_library()
    if not items:
        st.info("No videos yet. Upload an .mp4 above or add files to **ads/**.", icon="📂")
        return

    st.caption(f"{len(items)} video{'s' if len(items) != 1 else ''} · drag & drop to upload more")

    cols_per_row = 3
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(items):
                break
            item = items[idx]
            with col:
                _library_card(item, idx)


def _library_card(item: dict, idx: int) -> None:
    h = item["history_entry"]["hash"] if item["history_entry"] else None

    if item["is_analyzed"] and h:
        # Analyzed card — thumbnail + editable meta + scores
        thumb = history.thumbnail_path(h)
        if thumb.exists():
            st.image(str(thumb), use_container_width=True)
        else:
            st.markdown(
                '<div class="unanalyzed-thumb">🧠</div>',
                unsafe_allow_html=True,
            )

        name_key = f"lib_name_{h}"
        st.text_input(
            "Name",
            value=item["history_entry"]["name"],
            key=name_key,
            on_change=_lib_name_cb,
            args=(h, name_key),
            label_visibility="collapsed",
        )
        desc_key = f"lib_desc_{h}"
        st.text_area(
            "Description",
            value=item["history_entry"].get("description", ""),
            key=desc_key,
            on_change=_lib_desc_cb,
            args=(h, desc_key),
            height=64,
            placeholder="Add a note…",
            label_visibility="collapsed",
        )
        e = item["history_entry"]
        ts = e.get("timestamp", "")[:10]
        st.caption(
            f"Brain **{e['brain_score'] * 1000:.2f}** · "
            f"Impact **{e['impact_score'] * 1000:.2f}** · "
            f"Early **{e['early_attention_score'] * 1000:.2f}** · "
            f"{e['duration_trs'] * 2}s · {ts}"
        )
        if not Path(item["path"]).exists():
            st.caption("⚠ Video file unavailable")

    else:
        # Unanalyzed library card
        st.markdown(
            '<div class="unanalyzed-thumb">🎬</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{item['name']}**")
        st.caption("Not yet analyzed")
        if st.button("Analyze", key=f"analyze_{idx}", use_container_width=True):
            lib_path = item["path"]
            with st.spinner("Hashing video…"):
                video_hash = brain.hash_video(lib_path)
            with st.spinner("Loading model…"):
                model = get_model()
            analyze_video(lib_path, video_hash, model)
            st.rerun()


# -----------------------------------------------------------------------
# Compare tab
# -----------------------------------------------------------------------

def compare_tab_view() -> None:
    sel_a: dict | None = st.session_state.get("cmp_sel_a")
    sel_b: dict | None = st.session_state.get("cmp_sel_b")
    show_results: bool = st.session_state.get("cmp_show_results", False)

    if show_results:
        result_a = st.session_state.get("cmp_result_a")
        result_b = st.session_state.get("cmp_result_b")
        if result_a and result_b:
            _compare_results_view(result_a, result_b)
            return
        # Results missing — fall through to selection
        st.session_state.cmp_show_results = False

    # ── Selection view ──
    name_a = sel_a["name"] if sel_a else "— not selected —"
    name_b = sel_b["name"] if sel_b else "— not selected —"

    slot_a, slot_b, btn_col = st.columns([3, 3, 2])
    slot_a.markdown(f"**A:** {name_a}")
    slot_b.markdown(f"**B:** {name_b}")

    if btn_col.button(
        "Compare ▶",
        type="primary",
        use_container_width=True,
        disabled=not (sel_a and sel_b),
    ):
        result_a = load_result(sel_a["hash"], sel_a["video_path"])
        result_b = load_result(sel_b["hash"], sel_b["video_path"])
        if result_a and result_b:
            st.session_state.cmp_result_a = result_a
            st.session_state.cmp_result_b = result_b
            st.session_state.cmp_show_results = True
            st.rerun()
        else:
            st.error("Could not load results — cache may be missing.")

    st.divider()

    index = history.load_index()
    if not index:
        st.info("No analyzed ads yet. Go to **Library** to analyze some.", icon="📚")
        return

    st.caption("Select two ads below to compare")

    cols_per_row = 3
    for i in range(0, len(index), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(index):
                break
            with col:
                _compare_select_card(index[idx], idx)


def _compare_select_card(entry: dict, idx: int) -> None:
    sel_a = st.session_state.get("cmp_sel_a", {}) or {}
    sel_b = st.session_state.get("cmp_sel_b", {}) or {}
    is_a = sel_a.get("hash") == entry["hash"]
    is_b = sel_b.get("hash") == entry["hash"]

    thumb = history.thumbnail_path(entry["hash"])
    if thumb.exists():
        st.image(str(thumb), use_container_width=True)
    else:
        st.markdown('<div class="unanalyzed-thumb">🧠</div>', unsafe_allow_html=True)

    badges = []
    if is_a:
        badges.append("🔵 A")
    if is_b:
        badges.append("🔴 B")
    badge_str = "  ".join(badges)
    label = f"**{entry['name']}**" + (f"  {badge_str}" if badges else "")
    st.markdown(label)
    st.caption(
        f"Brain {entry['brain_score'] * 1000:.2f} · "
        f"Impact {entry['impact_score'] * 1000:.2f} · "
        f"{entry['duration_trs'] * 2}s"
    )

    b1, b2 = st.columns(2)
    if b1.button(
        "✓ A" if is_a else "Set A",
        key=f"cmp_seta_{idx}",
        use_container_width=True,
        type="primary" if is_a else "secondary",
    ):
        st.session_state.cmp_sel_a = entry
        st.session_state.cmp_show_results = False
        st.rerun()
    if b2.button(
        "✓ B" if is_b else "Set B",
        key=f"cmp_setb_{idx}",
        use_container_width=True,
        type="primary" if is_b else "secondary",
    ):
        st.session_state.cmp_sel_b = entry
        st.session_state.cmp_show_results = False
        st.rerun()


def _compare_results_view(result_a: dict, result_b: dict) -> None:
    sel_a = st.session_state.get("cmp_sel_a", {}) or {}
    sel_b = st.session_state.get("cmp_sel_b", {}) or {}
    name_a = sel_a.get("name", "Ad A")
    name_b = sel_b.get("name", "Ad B")

    if st.button("← Change selection", type="secondary"):
        st.session_state.cmp_show_results = False
        st.rerun()

    # Shared Y range so both engagement charts are directly comparable
    all_vals = (
        list(result_a["stats"]["engagement_over_time"])
        + list(result_b["stats"]["engagement_over_time"])
    )
    _ymin, _ymax = min(all_vals), max(all_vals)
    _margin = max((_ymax - _ymin) * 0.12, abs(_ymax) * 0.05, 1e-6)
    shared_yrange = (_ymin - _margin, _ymax + _margin)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"A — {name_a}")
        results_panel("a", result_a, color="rgb(59, 130, 246)", yrange=shared_yrange)
    with col_b:
        st.subheader(f"B — {name_b}")
        results_panel("b", result_b, color="rgb(239, 68, 68)", yrange=shared_yrange)

    comparison_section(result_a, result_b)
    region_legend()


# -----------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------

st.title("Ad Brain Analyzer")
st.caption(
    "Compare how two ads drive brain responses using "
    "[TRIBE v2](https://github.com/facebookresearch/tribev2) — Meta's fMRI foundation model."
)

lib_tab, cmp_tab = st.tabs(["📚 Library", "⚖️ Compare"])

with lib_tab:
    library_tab_view()

with cmp_tab:
    compare_tab_view()
