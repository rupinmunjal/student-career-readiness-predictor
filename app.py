import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src import MODEL_FEATURES
from src.data_pipeline import DATASET_SOURCES, DEFAULT_FEATURE_VALUES, DOMAIN_COLUMNS


ROLES = ["Web Dev", "Data", "DevOps", "Mobile"]
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _render_radar_chart(domain_scores: dict[str, float]) -> go.Figure:
    categories = list(domain_scores.keys())
    values = list(domain_scores.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Domain fit",
            line=dict(color="#2E86DE"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def _render_gauge(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Readiness (%)"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "#ffcccc"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 85,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def _readiness_label(score: float) -> tuple[str, str]:
    if score >= 70:
        return "High", "success"
    if score >= 40:
        return "Moderate", "warning"
    return "Low", "error"


@st.cache_resource
def initialize_components() -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.sidebar.warning(f"Could not connect to prediction API: {exc}")
        return {}


def prediction_page(model_info: dict[str, Any]) -> None:
    st.header("Career Readiness Prediction")

    if not model_info:
        st.error("No trained model available.")
        return

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Student Information")

        with st.expander("Academic Performance", expanded=True):
            c1, c2, c3 = st.columns(3)
            tenth_mark = c1.slider("10th Grade Marks (%)", 0.0, 100.0, 75.0)
            twelfth_mark = c2.slider("12th Grade Marks (%)", 0.0, 100.0, 70.0)
            college_mark = c3.slider("College Marks (%)", 0.0, 100.0, 80.0)

        with st.expander("Engagement & Junior Profile", expanded=True):
            c1, c2 = st.columns(2)
            cert_course = c1.selectbox("Certification Course", ["No", "Yes"])
            like_degree = c2.selectbox("Likes degree?", ["No", "Yes"])
            c1, c2 = st.columns(2)
            github_activity = c1.slider("GitHub activity (proxy 0-10)", 0.0, 10.0, 4.0)
            side_projects = c2.slider("Side projects count", 0.0, 10.0, 1.0)
            c1, c2 = st.columns(2)
            internship_exp = c1.slider("Internship experience (0/1)", 0.0, 1.0, 0.0, step=1.0)
            tech_rating = c2.slider("Tech skills self-rating (1-10)", 1.0, 10.0, 5.0)
            languages = st.slider("Languages known count", 0.0, 10.0, 2.0)

        with st.expander("Study Habits & Life Balance", expanded=True):
            study_time = st.selectbox(
                "Daily Study Time",
                [
                    "0-30 minutes",
                    "30-60 minutes",
                    "1-2 hours",
                    "2-3 hours",
                    "3-4 hours",
                    "More than 4 hours",
                ],
            )
            c1, c2, c3 = st.columns(3)
            travel_time = c1.selectbox(
                "Daily Travel Time",
                ["0-30 mins", "30-60 mins", "1-1.5 hrs", "1.5-2 hrs", "2-2.5 hrs", "2.5-3 hrs", "3+ hrs"],
            )
            stress_level = c2.selectbox("Stress Level", ["Bad", "Awful", "Good", "Fabulous"])
            part_time = c3.selectbox("Part-time Job", ["No", "Yes"])

    with col2:
        st.subheader("Prediction Results")

        study_time_map = {
            "0-30 minutes": 0.0, "30-60 minutes": 1.0, "1-2 hours": 2.0,
            "2-3 hours": 3.0, "3-4 hours": 4.0, "More than 4 hours": 5.0,
        }
        travel_map = {
            "0-30 mins": 0.0, "30-60 mins": 1.0, "1-1.5 hrs": 2.0,
            "1.5-2 hrs": 3.0, "2-2.5 hrs": 4.0, "2.5-3 hrs": 5.0, "3+ hrs": 6.0,
        }
        stress_map = {"Bad": 0.0, "Awful": 1.0, "Good": 2.0, "Fabulous": 3.0}

        input_data: dict[str, Any] = {
            "certification_course": 1.0 if cert_course == "Yes" else 0.0,
            "tenth_mark": float(tenth_mark),
            "twelfth_mark": float(twelfth_mark),
            "college_mark": float(college_mark),
            "daily_studying_time": study_time_map[study_time],
            "likes_degree": 1.0 if like_degree == "Yes" else 0.0,
            "travel_time": travel_map[travel_time],
            "stress_level": stress_map[stress_level],
            "part_time_job": 1.0 if part_time == "Yes" else 0.0,
            "github_activity_proxy": float(github_activity),
            "side_projects_count": float(side_projects),
            "internship_experience": float(internship_exp),
            "tech_skills_self_rating": float(tech_rating),
            "languages_known_count": float(languages),
        }

        if st.button("Predict Career Readiness", type="primary"):
            with st.spinner("Analyzing profile..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=input_data,
                        timeout=15,
                    )
                    response.raise_for_status()
                    profile = response.json()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            score = profile["readiness_score"]
            label, kind = _readiness_label(score)
            ci_low, ci_high = profile["confidence_interval"]

            g_col, r_col = st.columns([1, 1])
            g_col.plotly_chart(_render_gauge(score), use_container_width=True)
            with r_col:
                getattr(st, kind)(f"{label} readiness: **{score:.1f}%**")
                st.caption(f"Confidence interval: **{ci_low:.1f}% – {ci_high:.1f}%**")
                st.caption(f"Best-fit domain: **{profile['best_domain']}**")

            st.subheader("Domain Fit")
            st.plotly_chart(
                _render_radar_chart(profile["domain_scores"]),
                use_container_width=True,
            )

            st.subheader("Skill Gap Analysis")
            target = st.selectbox("Target role", ROLES, index=ROLES.index(profile["best_domain"]))
            gap_response = requests.post(
                f"{API_BASE_URL}/skill_gap",
                json={"student_profile": input_data, "target_role": target},
                timeout=15,
            )
            gap_response.raise_for_status()
            gap = gap_response.json()
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(f"**Met** ({len(gap['met'])})")
                if gap["met"]:
                    for item in gap["met"]:
                        st.markdown(f"- {item}")
                else:
                    st.caption("None yet")
            with g2:
                st.markdown(f"**Missing** ({len(gap['missing'])})")
                if gap["missing"]:
                    for item in gap["missing"]:
                        st.markdown(f"- {item}")
                else:
                    st.caption("None — fully aligned")

            importance = pd.DataFrame(model_info.get("feature_importance", []))
            if not importance.empty:
                st.subheader("Top 5 Feature Importances")
                fig = px.bar(
                    importance.head(5),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 5 features",
                )
                fig.update_layout(height=280, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Input summary"):
            st.dataframe(pd.DataFrame([input_data]).T, use_container_width=True)


def batch_upload_page(model_info: dict[str, Any]) -> None:
    st.header("Batch Upload")
    st.markdown(
        "Upload a CSV of student profiles to predict career readiness for many students at once. "
        f"Optional columns (missing ones are filled with sensible defaults): "
        f"`{', '.join(MODEL_FEATURES)}`."
    )

    if not model_info:
        st.error("No trained model available.")
        return

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Awaiting CSV upload.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if len(df) == 0:
        st.warning("Uploaded file has no rows.")
        return

    st.subheader("Uploaded data")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df)} rows, {len(df.columns)} columns")

    if not st.button("Run batch prediction", type="primary"):
        return

    with st.spinner("Running predictions..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/batch_predict",
                json={"students": df.to_dict(orient="records")},
                timeout=30,
            )
            response.raise_for_status()
            predictions = response.json()
            results = df.reset_index(drop=True).copy()
            results["predicted_junior_readiness"] = [
                item["readiness_score"] for item in predictions
            ]
            results["confidence_low"] = [item["confidence_interval"][0] for item in predictions]
            results["confidence_high"] = [item["confidence_interval"][1] for item in predictions]
            results["readiness_level"] = [item["readiness_level"] for item in predictions]
            results["best_domain"] = [item["best_domain"] for item in predictions]
            for domain_column, domain_label in zip(DOMAIN_COLUMNS, ROLES, strict=True):
                results[domain_column] = [
                    item["domain_scores"][domain_label] for item in predictions
                ]
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            return

    st.subheader("Results")
    st.dataframe(results.head(50), use_container_width=True)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv,
        file_name="batch_predictions.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Batch metrics")

    scores = results["predicted_junior_readiness"]
    mean_val = float(scores.mean())

    level_counts = results["readiness_level"].value_counts()
    n = len(results)
    high_pct = float(level_counts.get("High", 0)) / n * 100 if n else 0.0
    mod_pct = float(level_counts.get("Moderate", 0)) / n * 100 if n else 0.0
    low_pct = float(level_counts.get("Low", 0)) / n * 100 if n else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", f"{n}")
    k2.metric("Mean readiness", f"{mean_val:.1f}%")
    k3.metric("High (%)", f"{high_pct:.1f}%")
    k4.metric("Low (%)", f"{low_pct:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        domain_counts = results["best_domain"].value_counts().reindex(ROLES, fill_value=0)
        fig = px.bar(
            x=domain_counts.index,
            y=domain_counts.values,
            labels={"x": "Domain", "y": "Students"},
            title="Best-fit domain distribution",
        )
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(results, x="predicted_junior_readiness", nbins=20, title="Readiness distribution")
        fig.update_layout(height=320, xaxis_title="Readiness (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Level breakdown"):
        breakdown = pd.DataFrame(
            {
                "Level": ["High", "Moderate", "Low"],
                "Count": [
                    int(level_counts.get("High", 0)),
                    int(level_counts.get("Moderate", 0)),
                    int(level_counts.get("Low", 0)),
                ],
                "Percent": [high_pct, mod_pct, low_pct],
            }
        )
        st.dataframe(breakdown, use_container_width=True)


def analytics_page(model_info: dict[str, Any]) -> None:
    st.header("Analytics Dashboard")

    st.subheader("Training data sources")
    sources = pd.DataFrame(
        [
            {
                "Dataset": s["name"],
                "Rows used": s.get("rows_used", model_info.get("dataset_counts", {}).get(s["name"], 0)),
                "Target type": s["target_type"],
                "Target description": s["target_description"],
            }
            for s in DATASET_SOURCES
        ]
    )
    st.dataframe(sources, use_container_width=True)

    st.subheader("Dataset contribution")
    dataset_counts = model_info.get("dataset_counts", {})
    if dataset_counts:
        dc = pd.DataFrame(
            {"Dataset": list(dataset_counts.keys()), "Rows": list(dataset_counts.values())}
        )
        fig = px.bar(dc, x="Dataset", y="Rows", title="Rows per dataset")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model cross-validation scores")
    cv_results = model_info.get("cv_results", {})
    if cv_results:
        cv = pd.DataFrame(
            [
                {
                    "Model": name,
                    "CV MAE": f"{m['cv_mae']:.2f}",
                    "CV std": f"{m['cv_std']:.2f}",
                    "Test MAE": f"{m['test_mae']:.2f}",
                }
                for name, m in cv_results.items()
            ]
        ).sort_values("CV MAE")
        st.dataframe(cv, use_container_width=True)
        st.caption(
            f"Selected model: **{model_info.get('model_name')}** "
            f"(residual std = {model_info.get('residual_std', 0.0):.2f})"
        )

    importance = pd.DataFrame(model_info.get("feature_importance", []))
    if not importance.empty:
        st.subheader("Global feature importance (selected model)")
        fig = px.bar(
            importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 features",
        )
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def about_page() -> None:
    st.header("About This Application")

    st.markdown(
        """
This application predicts a student's career readiness for **junior/entry-level** software
engineering roles based on academic, engagement, and self-reported junior-profile signals.

### Pages
- **Prediction** — single-student analysis with readiness gauge, domain-fit radar chart,
  confidence interval, and skill-gap analysis for a target role.
- **Batch Upload** — upload a CSV of many students, view results, and download predictions with
  aggregate batch metrics (mean readiness, level distribution, domain distribution).
- **Analytics** — training-data provenance, cross-validation scores, and global feature
  importance for the selected model.
- **About** (this page).

### Model
- 6 candidate models are trained (Random Forest, Gradient Boosting, SVR, Linear Regression,
  XGBoost, LightGBM when available) and the lowest-MAE model is auto-selected.
- 5-fold cross-validation is used for ranking; a residual-based 95% confidence interval is
  reported for each prediction.
- Features include 14 raw/canonical inputs plus 3 engineered relevance scores and 4 domain-fit
  scores (Web Dev, Data, DevOps, Mobile).

### Data
The model is trained on **450 honestly-aligned records** from two publicly available datasets:
self-reported student attitude/behavior surveys and real campus placement outcomes. No
synthetic or fabricated targets are used.

### How to run
```bash
uv sync
uv run streamlit run app.py
```
"""
    )
    st.markdown("Built with Streamlit, scikit-learn, Plotly.")


def main() -> None:
    st.set_page_config(
        page_title="Student Career Readiness Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎓 Student Career Readiness Predictor")
    st.markdown("---")

    model_info = initialize_components()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Prediction", "Batch Upload", "Analytics", "About"])

    if page == "Prediction":
        prediction_page(model_info)
    elif page == "Batch Upload":
        batch_upload_page(model_info)
    elif page == "Analytics":
        analytics_page(model_info)
    else:
        about_page()


if __name__ == "__main__":
    main()
