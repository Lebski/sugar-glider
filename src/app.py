"""
Ad Brain Analyzer — Streamlit UI.

Compares brain responses to two ad videos using TRIBE v2.
Run with: uv run streamlit run src/app.py
"""

import functools
import hashlib
import http.server
import os
import threading
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import brain
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

    /* Force sidebar to always be light */
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        color: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #111827 !important;
    }
    /* Selectbox dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
        color: #111827 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #111827 !important;
    }
    /* Radio buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #111827 !important;
    }
    /* File uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: #111827 !important;
    }
    /* Divider */
    [data-testid="stSidebar"] hr { border-color: #e5e7eb; }

    div[data-testid="stMetric"] { background: #f9fafb; border-radius: 8px; padding: 0.75rem 1rem; }
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
    server = http.server.HTTPServer(("127.0.0.1", FILE_SERVER_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
        # Path outside project root — fall back to st.video
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


def run_and_cache(label: str, video_path: str, video_hash: str, model) -> dict | None:
    """
    Three-layer cache:
      1. st.session_state  — instant, lives for the browser session
      2. outputs/client_cache/<hash>/  — survives refreshes and new sessions
      3. API call  — only when neither cache has the result
    """
    cache_key = f"result_{label}"

    # Layer 1: session state
    if cache_key in st.session_state:
        if st.session_state[cache_key].get("video_hash") == video_hash:
            return st.session_state[cache_key]

    # Layer 2: disk cache
    cached = result_cache.load(video_hash)
    if cached is not None:
        cached["video_path"] = video_path
        cached["video_hash"] = video_hash
        st.session_state[cache_key] = cached
        return cached

    # Layer 3: run inference — wrap in a placeholder so we can clear it when done
    placeholder = st.empty()
    with placeholder.container():
        with st.status(
            f"Analyzing Ad {label.upper()} — this may take a few minutes...",
            expanded=True,
        ) as status:
            st.write("Extracting features from video...")
            preds, segments = brain.run_inference(video_path, model)
            st.write("Computing brain statistics...")
            stats = brain.compute_stats(preds, segments)
            st.write("Rendering brain map...")
            png_path = f"outputs/renders/{video_hash}_mean.png"
            brain.render_brain_png(stats["preds_mean"], png_path)
            st.write("Saving to cache...")
            result_cache.save(video_hash, stats, Path(png_path).read_bytes())
            status.update(label=f"Ad {label.upper()} ready.", state="complete")
    placeholder.empty()

    result = {
        "video_path": video_path,
        "video_hash": video_hash,
        "stats": stats,
        "brain_png": str(result_cache._dir(video_hash) / "brain.png"),
    }
    st.session_state[cache_key] = result
    return result


# -----------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------

def engagement_chart(stats: dict, color: str, cursor_idx: int | None = None) -> go.Figure:
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
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
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

def results_panel(label: str, result: dict, color: str):
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
        # Average metrics — values scaled ×1000 (milli-BOLD) to fit metric boxes
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brain Score", f"{stats['overall_score'] * 1000:.2f}", help="Mean predicted BOLD activation ×10⁻³")
        early = stats.get("early_attention_score")
        c2.metric("Early Attn", f"{early * 1000:.2f}" if early is not None else "—", help="Log-weighted score — first seconds count more (×10⁻³ BOLD)")
        c3.metric("Peak Moment", f"{stats['peak_timestamp_s']:.0f}s")
        c4.metric("Duration", f"{len(stats['engagement_over_time'])} TRs")

        # Live / current-segment stats
        seg_ts = stats["segment_timestamps"]
        seg_act = stats["engagement_over_time"]
        mean_act = float(seg_act.mean())
        current_val = float(seg_act[seg_idx]) if seg_idx < len(seg_act) else mean_act
        per_seg_rois = stats.get("per_segment_top_rois", [])
        top_rois_now = per_seg_rois[seg_idx] if seg_idx < len(per_seg_rois) else []

        ts_label = f"{seg_ts[seg_idx]:.0f}s" if seg_idx < len(seg_ts) else "0s"
        st.markdown(f"**Now · {ts_label}**")
        lc1, lc2 = st.columns(2)
        lc1.metric(
            "Now (×10⁻³)",
            f"{current_val * 1000:.2f}",
            delta=f"{(current_val - mean_act) * 1000:+.2f} vs mean",
        )
        roi_help = "\n".join(
            f"{r}: {HCP_ROI_NAMES.get(r, 'HCP cortical area')}" for r in top_rois_now
        ) if top_rois_now else None
        lc2.metric("Active regions", " · ".join(top_rois_now) if top_rois_now else "—", help=roi_help)

        st.markdown("**Engagement over time**")
        st.plotly_chart(
            engagement_chart(stats, color, cursor_idx=seg_idx),
            use_container_width=True, key=f"eng_{label}",
        )

        st.markdown("**Top activated regions**")
        st.plotly_chart(roi_bar_chart(stats, color), use_container_width=True, key=f"roi_{label}")

        with st.expander("Cognitive breakdown"):
            ca, cb, cc = st.columns(3)
            ca.metric("Visual cortex", f"{stats['visual_score'] * 1000:.2f}", help="V1 · V2 · V3 · V4 · MT · MST · V3A · V3B\nHow strongly the visual creative is being processed — motion, colour, objects")
            cb.metric("Language cortex", f"{stats['language_score'] * 1000:.2f}", help="Broca (BA44/45) · Superior Temporal Sulcus (STS) · TE1a · TE1m\nHow much spoken or written language is being processed")
            cc.metric("Attention network", f"{stats['attention_score'] * 1000:.2f}", help="FEF · IPS1 · VIP · LIPv · LIPd · 7PC\nHow strongly top-down attention and gaze control are engaged")


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
            with st.spinner("Rendering difference map..."):
                brain.render_delta_brain_png(stats_a["preds_mean"], stats_b["preds_mean"], delta_png)
        if Path(delta_png).exists():
            st.image(delta_png, use_container_width=True)
            st.caption("Red = Ad B more active · Blue = Ad A more active")


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------

def ad_selector(slot: str) -> tuple[str, str] | tuple[None, None]:
    """Returns (video_path, video_hash) or (None, None)."""
    st.sidebar.markdown(f"**Ad {slot.upper()}**")
    mode = st.sidebar.radio(
        "Source", ["Library", "Upload"],
        key=f"mode_{slot}", horizontal=True, label_visibility="collapsed",
    )
    if mode == "Library":
        names = library.get_library_names()
        if not names:
            st.sidebar.warning("No ads found — add .mp4 files to **ads/**.")
            return None, None
        selected = st.sidebar.selectbox(
            "Ad", names, key=f"lib_{slot}", label_visibility="collapsed"
        )
        path = library.get_library_path(selected)
        return path, brain.hash_video(path)
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload .mp4", type=["mp4"], key=f"upload_{slot}",
            label_visibility="collapsed",
        )
        return save_upload(uploaded)


# -----------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------

st.title("Ad Brain Analyzer")
st.caption(
    "Compare how two ads drive brain responses using "
    "[TRIBE v2](https://github.com/facebookresearch/tribev2) — Meta's fMRI foundation model."
)

st.sidebar.title("Select Ads")
path_a, hash_a = ad_selector("a")
st.sidebar.markdown("---")
path_b, hash_b = ad_selector("b")
st.sidebar.markdown("---")
compare_clicked = st.sidebar.button(
    "Compare",
    type="primary",
    use_container_width=True,
    disabled=(not path_a or not path_b),
)

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------

if compare_clicked and path_a and path_b:
    with st.spinner("Loading TRIBE v2 model..."):
        model = get_model()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Ad A")
        result_a = run_and_cache("a", path_a, hash_a, model)
    with col_b:
        st.subheader("Ad B")
        result_b = run_and_cache("b", path_b, hash_b, model)

    if result_a and result_b:
        with col_a:
            results_panel("a", result_a, color="rgb(59, 130, 246)")
        with col_b:
            results_panel("b", result_b, color="rgb(239, 68, 68)")
        comparison_section(result_a, result_b)

elif "result_a" in st.session_state and "result_b" in st.session_state:
    result_a = st.session_state["result_a"]
    result_b = st.session_state["result_b"]
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Ad A")
        results_panel("a", result_a, color="rgb(59, 130, 246)")
    with col_b:
        st.subheader("Ad B")
        results_panel("b", result_b, color="rgb(239, 68, 68)")
    comparison_section(result_a, result_b)

else:
    st.info("Select two ads in the sidebar and click **Compare** to see brain responses.", icon="👈")
