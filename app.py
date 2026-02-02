from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import StudentCareerWillingnessPredictor, MODEL_FEATURES
from src.data_pipeline import DATASET_SOURCES, DEFAULT_FEATURE_VALUES, DOMAIN_COLUMNS


ROLES = ["Web Dev", "Data", "DevOps", "Mobile"]


import os

@st.cache_resource
def initialize_components():
    predictor = StudentCareerWillingnessPredictor()

    model_path = 'models/student_career_willingness_model.pkl'
    if os.path.exists(model_path):
        try:
            predictor.load_model(model_path)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load pre-trained model: {e}")
            predictor.train("data/Student Attitude and Behavior.csv")
            predictor.save_model(model_path)
    else:
        predictor.train("data/Student Attitude and Behavior.csv")
        predictor.save_model(model_path)

    return predictor


def prediction_page(predictor):
    st.header("🔮 Career Willingness Prediction")

    if predictor.model is None:
        st.error("⚠️ No trained model available. Please check default dataset.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Student Information")

        with st.expander("📚 Academic Performance", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                tenth_mark = st.slider("10th Grade Marks (%)", 0.0, 100.0, 75.0)
            with col_b:
                twelfth_mark = st.slider("12th Grade Marks (%)", 0.0, 100.0, 70.0)
            with col_c:
                college_mark = st.slider("College Marks (%)", 0.0, 100.0, 80.0)

        with st.expander("💡 Engagement & Interest", expanded=True):
            col_d, col_e = st.columns(2)
            with col_d:
                cert_course = st.selectbox("Certification Course", ["No", "Yes"])
            with col_e:
                like_degree = st.selectbox("Do you like your degree?", ["No", "Yes"])

        with st.expander("📖 Study Habits", expanded=True):
            study_time = st.selectbox(
                "Daily Study Time",
                ["0-30 minutes", "30-60 minutes", "1-2 hours", "2-3 hours", "3-4 hours", "More than 4 hours"]
            )

        with st.expander("⚖️ Life Balance", expanded=True):
            col_f, col_g, col_h = st.columns(3)
            with col_f:
                travel_time = st.selectbox(
                    "Daily Travel Time",
                    ["0-30 mins", "30-60 mins", "1-1.5 hrs", "1.5-2 hrs", "2-2.5 hrs", "2.5-3 hrs", "3+ hrs"]
                )
            with col_g:
                stress_level = st.selectbox("Stress Level", ["Bad", "Awful", "Good", "Fabulous"])
            with col_h:
                part_time = st.selectbox("Part-time Job", ["No", "Yes"])

    with col2:
        st.subheader("📊 Prediction Results")

        study_time_mapping = {
            "0-30 minutes": 0.0, "30-60 minutes": 1.0, "1-2 hours": 2.0,
            "2-3 hours": 3.0, "3-4 hours": 4.0, "More than 4 hours": 5.0
        }

        travel_mapping = {
            "0-30 mins": 0.0, "30-60 mins": 1.0, "1-1.5 hrs": 2.0,
            "1.5-2 hrs": 3.0, "2-2.5 hrs": 4.0, "2.5-3 hrs": 5.0, "3+ hrs": 6.0
        }

        stress_mapping = {"Bad": 0.0, "Awful": 1.0, "Good": 2.0, "Fabulous": 3.0}

        input_data = {
            'Certification Course': 1.0 if cert_course == "Yes" else 0.0,
            '10th Mark': float(tenth_mark),
            '12th Mark': float(twelfth_mark),
            'college mark': float(college_mark),
            'daily studing time': study_time_mapping[study_time],
            'Do you like your degree?': 1.0 if like_degree == "Yes" else 0.0,
            'Travelling Time ': travel_mapping[travel_time],
            'Stress Level ': stress_mapping[stress_level],
            'part-time job': 1.0 if part_time == "Yes" else 0.0
        }

        if st.button("🎯 Predict Career Willingness", type="primary"):
            with st.spinner("Analyzing student profile..."):
                try:
                    prediction = predictor.predict(input_data)

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Career Willingness (%)"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    if prediction >= 75:
                        st.success(f"🌟 High Career Willingness ({prediction:.1f}%)")
                    elif prediction >= 50:
                        st.warning(f"⚖️ Moderate Career Willingness ({prediction:.1f}%)")
                    else:
                        st.error(f"⚠️ Low Career Willingness ({prediction:.1f}%)")

                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        st.subheader("📋 Input Summary")
        summary_df = pd.DataFrame([input_data]).T
        summary_df.columns = ['Value']
        st.dataframe(summary_df, use_container_width=True)


def batch_upload_page(predictor):
    st.header("Batch Upload")
    st.markdown(
        "Upload a CSV of student profiles to predict career readiness for many students at once. "
        f"Optional columns (missing ones are filled with sensible defaults): "
        f"`{', '.join(MODEL_FEATURES)}`."
    )

    if predictor.model is None:
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
            results = predictor.predict_batch(df)
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


def analytics_page(predictor):
    st.header("Analytics Dashboard")

    st.subheader("Training data sources")
    sources = pd.DataFrame(
        [
            {
                "Dataset": s["name"],
                "Rows used": s.get("rows_used", predictor.dataset_counts.get(s["name"], 0)),
                "Target type": s["target_type"],
                "Target description": s["target_description"],
            }
            for s in DATASET_SOURCES
        ]
    )
    st.dataframe(sources, use_container_width=True)

    st.subheader("Dataset contribution")
    if predictor.dataset_counts:
        dc = pd.DataFrame(
            {"Dataset": list(predictor.dataset_counts.keys()), "Rows": list(predictor.dataset_counts.values())}
        )
        fig = px.bar(dc, x="Dataset", y="Rows", title="Rows per dataset")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset counts unavailable (model loaded without retraining).")

    st.subheader("Model cross-validation scores")
    if predictor.cv_results:
        cv = pd.DataFrame(
            [
                {
                    "Model": name,
                    "CV MAE": f"{m['cv_mae']:.2f}",
                    "CV std": f"{m['cv_std']:.2f}",
                    "Test MAE": f"{m['test_mae']:.2f}",
                }
                for name, m in predictor.cv_results.items()
            ]
        ).sort_values("CV MAE")
        st.dataframe(cv, use_container_width=True)
        st.caption(f"Selected model: **{predictor.model_name}** (residual std = {predictor.residual_std:.2f})")
    else:
        st.info("Cross-validation results unavailable.")

    importance = predictor.get_feature_importance()
    if importance is not None:
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
    else:
        st.info("Feature importance not available for this model type.")


def about_page():
    st.header("About This Application")

    st.markdown(
        """
This application predicts a student's career readiness for **junior/entry-level** software
engineering roles based on academic, engagement, and self-reported junior-profile signals.

### Pages
- **Prediction** — single-student analysis with a readiness gauge and domain-fit assessment.
- **Batch Upload** — upload a CSV of many students, view results, and download predictions with
  aggregate batch metrics.
- **Analytics** — training-data provenance, cross-validation scores, and global feature importance.
- **About** (this page).

### Model
- 6 candidate models are trained (Random Forest, Gradient Boosting, SVR, Linear Regression,
  XGBoost, LightGBM when available) and the lowest-MAE model is auto-selected.
- 5-fold cross-validation is used for ranking; a residual-based 95% confidence interval is
  reported for each prediction.
- Features include 14 raw inputs plus 3 engineered relevance scores and 4 domain-fit scores
  (Web Dev, Data, DevOps, Mobile).

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


def main():
    st.set_page_config(
        page_title="Student Career Readiness Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎓 Student Career Readiness Predictor")
    st.markdown("---")

    predictor = initialize_components()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Prediction", "Batch Upload", "Analytics", "About"])

    if page == "Prediction":
        prediction_page(predictor)
    elif page == "Batch Upload":
        batch_upload_page(predictor)
    elif page == "Analytics":
        analytics_page(predictor)
    else:
        about_page()


if __name__ == "__main__":
    main()
