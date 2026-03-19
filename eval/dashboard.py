"""
SynCVE Evaluation Dashboard
Run: streamlit run eval/dashboard.py
"""

import json
import os
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path(__file__).parent / "results"

st.set_page_config(
    page_title="SynCVE Eval Dashboard",
    layout="wide",
    page_icon="\U0001f9e0",
)

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #1a1a2e; color: #ffffff; }
    .stMetric label { color: #cccccc !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stMetric [data-testid="stMetricDelta"] svg { display: none; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    .stDataFrame { color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] button { color: #cccccc; }
    .stTabs [aria-selected="true"] { color: #E69F00 !important; border-bottom-color: #E69F00 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(filename: str) -> dict | None:
    """Load a JSON file from the results directory (supports subdirs).

    ``filename`` is relative to RESULTS_DIR, e.g. ``"baseline/fer2013_retinaface.json"``.
    """
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def img_path(filename: str) -> Path:
    """Return full path for an image relative to RESULTS_DIR."""
    return RESULTS_DIR / filename


def show_image_or_placeholder(filename: str, caption: str = ""):
    """Display a PNG from results dir, or a placeholder message."""
    p = img_path(filename)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not found: {filename}. Run the benchmark first.")


def format_pct(val, digits=2):
    if val is None:
        return "N/A"
    return f"{val * 100:.{digits}f}%"


def format_float(val, digits=4):
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def delta_str(val):
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.4f}"


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("SynCVE Evaluation Dashboard")
st.caption("Interactive viewer for all evaluation results")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_baseline, tab_ablation, tab_pipeline, tab_temporal = st.tabs(
    ["Overview", "Baseline", "Ablation", "Pipeline", "Temporal"]
)


# ============================= OVERVIEW =====================================
with tab_overview:
    st.header("Experiment Overview")

    json_files = sorted(RESULTS_DIR.rglob("*.json")) if RESULTS_DIR.exists() else []
    if not json_files:
        st.warning("No result JSON files found in eval/results/. Run benchmarks first.")
    else:
        rows = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # Show path relative to RESULTS_DIR for clarity
            name = str(jf.relative_to(RESULTS_DIR)).replace("\\", "/").removesuffix(".json")
            meta = data.get("metadata", {})
            ts = meta.get("timestamp", "N/A")

            # Try to extract key metrics regardless of structure
            accuracy = (
                data.get("overall_accuracy")
                or data.get("accuracy")
            )
            weighted_f1 = None
            cr = data.get("classification_report", {})
            if "weighted avg" in cr:
                weighted_f1 = cr["weighted avg"].get("f1-score")

            rows.append(
                {
                    "Experiment": name,
                    "Accuracy": format_float(accuracy) if accuracy is not None else "—",
                    "Weighted F1": format_float(weighted_f1) if weighted_f1 is not None else "—",
                    "Timestamp": ts,
                }
            )

        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Quick Metrics")
        # Show FER2013 baseline quick stats
        fer = load_json("baseline/fer2013_retinaface.json")
        raf = load_json("baseline/rafdb_retinaface.json")

        col1, col2, col3, col4 = st.columns(4)
        if fer:
            col1.metric("FER2013 Accuracy", format_pct(fer.get("overall_accuracy")))
            cr_f = fer.get("classification_report", {})
            col2.metric(
                "FER2013 Weighted F1",
                format_float(cr_f.get("weighted avg", {}).get("f1-score")),
            )
        else:
            col1.metric("FER2013 Accuracy", "N/A")
            col2.metric("FER2013 Weighted F1", "N/A")

        if raf:
            col3.metric("RAF-DB Accuracy", format_pct(raf.get("overall_accuracy")))
            cr_r = raf.get("classification_report", {})
            col4.metric(
                "RAF-DB Weighted F1",
                format_float(cr_r.get("weighted avg", {}).get("f1-score")),
            )
        else:
            col3.metric("RAF-DB Accuracy", "N/A")
            col4.metric("RAF-DB Weighted F1", "N/A")


# ============================= BASELINE =====================================
with tab_baseline:
    st.header("Baseline Results")

    for dataset_tag, json_name, prefix, detector in [
        ("FER2013", "baseline/fer2013_retinaface.json", "fer2013", "retinaface"),
        ("RAF-DB", "baseline/rafdb_retinaface.json", "rafdb", "retinaface"),
    ]:
        st.subheader(dataset_tag)
        data = load_json(json_name)

        if data is None:
            st.warning(f"No results for {dataset_tag}. Run `python -m eval.benchmark_{prefix}` first.")
            continue

        # Key metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", format_pct(data.get("overall_accuracy")))
        cr = data.get("classification_report", {})
        c2.metric("Weighted F1", format_float(cr.get("weighted avg", {}).get("f1-score")))
        c3.metric("Detection Rate", format_pct(data.get("detection_rate")))
        lat = data.get("latency", {})
        c4.metric("Mean Latency", f'{lat.get("mean_ms", 0):.1f} ms')

        # Plots
        col_left, col_right = st.columns(2)
        with col_left:
            show_image_or_placeholder(f"baseline/plots/{prefix}_{detector}_confusion_matrix.png", f"{dataset_tag} Confusion Matrix")
            show_image_or_placeholder(f"baseline/plots/{prefix}_{detector}_per_class_metrics.png", f"{dataset_tag} Per-Class Metrics")
        with col_right:
            show_image_or_placeholder(f"baseline/plots/{prefix}_{detector}_roc_curves.png", f"{dataset_tag} ROC Curves")
            show_image_or_placeholder(f"baseline/plots/{prefix}_{detector}_latency_histogram.png", f"{dataset_tag} Latency Distribution")

        st.divider()


# ============================= ABLATION =====================================
with tab_ablation:
    st.header("Ablation Studies")

    import pandas as pd

    # --- Preprocessing ablation ---
    st.subheader("Preprocessing Ablation")
    preprocess_data = load_json("ablation/preprocess.json")
    if preprocess_data:
        configs = preprocess_data.get("configs", {})
        rows = []
        for cid, cfg in configs.items():
            lat = cfg.get("latency", {})
            rows.append(
                {
                    "Config": cid,
                    "Accuracy": format_pct(cfg.get("accuracy")),
                    "Weighted F1": format_float(cfg.get("weighted_f1")),
                    "Detection Rate": format_pct(cfg.get("detection_rate")),
                    "Mean Latency (ms)": f'{lat.get("mean_ms", 0):.1f}',
                    "Failures": cfg.get("num_failures", 0),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        show_image_or_placeholder("ablation/plots/preprocess_comparison.png", "Preprocessing Ablation Comparison")
    else:
        st.warning("Run `python -m eval.ablation_preprocess` first.")

    st.divider()

    # --- Detector ablation ---
    st.subheader("Detector Ablation")
    detector_data = load_json("ablation/detector.json")
    if detector_data:
        detectors = detector_data.get("detectors", {})
        rows = []
        for det_name, det_info in detectors.items():
            lat = det_info.get("latency", {})
            rows.append(
                {
                    "Detector": det_name,
                    "Accuracy": format_pct(det_info.get("accuracy")),
                    "Weighted F1": format_float(det_info.get("weighted_f1")),
                    "Detection Rate": format_pct(det_info.get("detection_rate")),
                    "Mean Latency (ms)": f'{lat.get("mean_ms", 0):.1f}',
                    "P95 Latency (ms)": f'{lat.get("p95_ms", 0):.1f}',
                    "Failures": det_info.get("num_failures", 0),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        show_image_or_placeholder("ablation/plots/detector_comparison.png", "Detector Ablation — Accuracy vs Latency")
    else:
        st.warning("Run `python -m eval.ablation_detector` first.")

    st.divider()

    # --- Postprocess ablation ---
    st.subheader("Post-Processing Ablation")
    postprocess_data = load_json("ablation/postprocess.json")
    if postprocess_data:
        configs = postprocess_data.get("configs", {})
        rows = []
        for cid, cfg in configs.items():
            settings = cfg.get("settings", {})
            rows.append(
                {
                    "Config": cid,
                    "EMA Alpha": settings.get("ema_alpha", "—"),
                    "Noise Floor": settings.get("noise_floor", "—"),
                    "Consistency": format_float(cfg.get("consistency_score")),
                    "Flicker Rate": f'{cfg.get("flicker_rate", 0):.1f}',
                    "Accuracy": format_pct(cfg.get("accuracy")),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        show_image_or_placeholder("ablation/plots/postprocess_comparison.png", "Post-Processing Ablation Comparison")
    else:
        st.warning("Run `python -m eval.ablation_postprocess` first.")


# ============================= PIPELINE =====================================
with tab_pipeline:
    st.header("Full Pipeline vs Baseline")

    pipeline_data = load_json("pipeline/pipeline_vs_b0.json")
    if pipeline_data is None:
        st.warning("Run `python -m eval.pipeline_vs_baseline` first.")
    else:
        comparisons = pipeline_data.get("comparisons", {})

        for ds_name, comp in comparisons.items():
            st.subheader(ds_name)

            b0 = comp.get("b0", {})
            pipe = comp.get("pipeline", {})
            delta = comp.get("delta_vs_b0", {})

            # Metric cards with delta highlighting
            c1, c2, c3, c4 = st.columns(4)

            pipe_acc = pipe.get("accuracy")
            base_acc = b0.get("accuracy")
            delta_acc = delta.get("accuracy")
            c1.metric(
                "Accuracy",
                format_pct(pipe_acc),
                delta=delta_str(delta_acc) if delta_acc is not None else None,
            )

            pipe_f1 = pipe.get("weighted_f1")
            base_f1 = b0.get("weighted_f1")
            delta_f1 = delta.get("weighted_f1")
            c2.metric(
                "Weighted F1",
                format_float(pipe_f1),
                delta=delta_str(delta_f1) if delta_f1 is not None else None,
            )

            c3.metric(
                "Detection Rate",
                format_pct(pipe.get("detection_rate")),
            )

            pipe_lat = pipe.get("latency", {})
            base_lat = b0.get("latency", {})
            base_lat_mean = base_lat.get("mean_ms", 0)
            pipe_lat_mean = pipe_lat.get("mean_ms", 0)
            lat_delta = pipe_lat_mean - base_lat_mean if pipe_lat_mean and base_lat_mean else None
            c4.metric(
                "Mean Latency (ms)",
                f"{pipe_lat_mean:.1f}" if pipe_lat_mean else "N/A",
                delta=f"{lat_delta:+.1f} ms" if lat_delta is not None else None,
                delta_color="inverse",
            )

            # Comparison table
            import pandas as pd

            rows = [
                {
                    "Metric": "Accuracy",
                    "B0 Baseline": format_float(base_acc),
                    "Pipeline": format_float(pipe_acc),
                    "Delta": delta_str(delta_acc),
                },
                {
                    "Metric": "Weighted F1",
                    "B0 Baseline": format_float(base_f1),
                    "Pipeline": format_float(pipe_f1),
                    "Delta": delta_str(delta_f1),
                },
                {
                    "Metric": "Detection Rate",
                    "B0 Baseline": format_pct(b0.get("detection_rate")),
                    "Pipeline": format_pct(pipe.get("detection_rate")),
                    "Delta": "—",
                },
                {
                    "Metric": "Mean Latency (ms)",
                    "B0 Baseline": f"{base_lat_mean:.1f}",
                    "Pipeline": f"{pipe_lat_mean:.1f}" if pipe_lat_mean else "N/A",
                    "Delta": f"{lat_delta:+.1f}" if lat_delta is not None else "—",
                },
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Pipeline config info
            with st.expander("Pipeline Configuration"):
                st.json(pipe.get("config", {}))

            st.divider()

        # Global pipeline config
        cfg = pipeline_data.get("pipeline_config", {})
        if cfg:
            with st.expander("Global Pipeline Configuration"):
                st.json(cfg)

        show_image_or_placeholder("pipeline/plots/comparison.png", "Pipeline vs Baseline Comparison")


# ============================= TEMPORAL =====================================
with tab_temporal:
    st.header("Temporal Post-Processing Analysis")

    postprocess_data = load_json("ablation/postprocess.json")
    if postprocess_data is None:
        st.warning("Run `python -m eval.ablation_postprocess` first.")
    else:
        import pandas as pd

        configs = postprocess_data.get("configs", {})

        # Build comparison dataframe
        rows = []
        for cid, cfg in configs.items():
            rows.append(
                {
                    "Config": cid,
                    "Consistency": cfg.get("consistency_score", 0),
                    "Flicker Rate": cfg.get("flicker_rate", 0),
                    "Accuracy": cfg.get("accuracy", 0),
                }
            )
        df = pd.DataFrame(rows)

        # Three metric columns
        c1, c2, c3 = st.columns(3)

        # Best consistency
        best_cons = df.loc[df["Consistency"].idxmax()]
        c1.metric("Best Consistency", f'{best_cons["Consistency"]:.4f}', delta=best_cons["Config"])

        # Lowest flicker
        best_flicker = df.loc[df["Flicker Rate"].idxmin()]
        c2.metric("Lowest Flicker Rate", f'{best_flicker["Flicker Rate"]:.1f}', delta=best_flicker["Config"])

        # Best accuracy
        best_acc = df.loc[df["Accuracy"].idxmax()]
        c3.metric("Best Accuracy", format_pct(best_acc["Accuracy"]), delta=best_acc["Config"])

        st.subheader("Consistency / Flicker / Accuracy by Config")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar charts using Streamlit native charts
        st.subheader("Consistency Score")
        chart_df = df.set_index("Config")
        st.bar_chart(chart_df[["Consistency"]])

        st.subheader("Flicker Rate (lower is better)")
        st.bar_chart(chart_df[["Flicker Rate"]])

        st.subheader("Accuracy")
        st.bar_chart(chart_df[["Accuracy"]])

        show_image_or_placeholder("ablation/plots/postprocess_comparison.png", "Post-Processing Ablation Plot")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("SynCVE Evaluation Dashboard | Results directory: " + str(RESULTS_DIR.resolve()))
