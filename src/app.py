"""
Ad Brain Analyzer — Streamlit UI.

Compares brain responses to two ad videos using TRIBE v2.
Run with: uv run streamlit run src/app.py
"""

import hashlib
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import brain
import library
import result_cache

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

def engagement_chart(stats: dict, color: str) -> go.Figure:
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
    if result.get("video_path") and Path(result["video_path"]).exists():
        st.video(result["video_path"])
    brain_tab, stats_tab = st.tabs(["Brain Map", "Statistics"])

    with brain_tab:
        if Path(result["brain_png"]).exists():
            st.image(result["brain_png"], use_container_width=True)
        else:
            st.caption("Brain map not available.")

    with stats_tab:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brain Score", f"{stats['overall_score']:.5f}")
        c2.metric("Early Attention", f"{stats['early_attention_score']:.5f}", help="Log-weighted score — first seconds count more")
        c3.metric("Peak Moment", f"{stats['peak_timestamp_s']:.0f}s")
        c4.metric("Duration", f"{len(stats['engagement_over_time'])} TRs")

        st.markdown("**Engagement over time**")
        st.plotly_chart(engagement_chart(stats, color), use_container_width=True, key=f"eng_{label}")

        st.markdown("**Top activated regions**")
        st.plotly_chart(roi_bar_chart(stats, color), use_container_width=True, key=f"roi_{label}")

        with st.expander("Cognitive breakdown"):
            ca, cb, cc = st.columns(3)
            ca.metric("Visual cortex", f"{stats['visual_score']:.5f}", help="V1 · V2 · V3 · MT")
            cb.metric("Language cortex", f"{stats['language_score']:.5f}", help="Broca · Wernicke")
            cc.metric("Attention network", f"{stats['attention_score']:.5f}", help="FEF · IPS · VIP")


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
