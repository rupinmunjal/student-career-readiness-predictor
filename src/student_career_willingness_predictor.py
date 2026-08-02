from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data_pipeline import (
    DEFAULT_FEATURE_VALUES,
    DOMAIN_COLUMNS,
    MODEL_FEATURES,
    TARGET_COLUMN,
    normalize_input_frame,
    prepare_training_dataset,
)


class StudentCareerWillingnessPredictor:
    def __init__(self, load_on_init: bool = True) -> None:
        self.model: Any | None = None
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = MODEL_FEATURES.copy()
        self.model_name: str | None = None
        self.model_type: str = "raw"
        self.residual_std: float = 8.0
        self.cv_results: dict[str, dict[str, float]] = {}
        self.training_reference: pd.DataFrame | None = None
        self.dataset_counts: dict[str, int] = {}
        self.used_files: list[str] = []
        if load_on_init:
            self._load_configured_model()

    def _load_configured_model(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            try:
                self._load_from_mlflow(tracking_uri)
                return
            except Exception:
                pass
        local_path = Path("models/student_career_willingness_model.pkl")
        if local_path.exists():
            self.load_model(str(local_path))

    def _load_from_mlflow(self, tracking_uri: str) -> None:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        version = MlflowClient().get_model_version_by_alias("student-readiness-model", "latest")
        bundle_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{version.run_id}/predictor_bundle.pkl"
        )
        self.load_model(bundle_path)

    def _build_candidate_models(self) -> dict[str, tuple[Any, str]]:
        models: dict[str, tuple[Any, str]] = {
            "Random Forest": (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                ),
                "raw",
            ),
            "Gradient Boosting": (
                GradientBoostingRegressor(n_estimators=250, learning_rate=0.06, random_state=42),
                "raw",
            ),
            "SVR": (SVR(kernel="rbf", C=5.0, gamma="scale"), "scaled"),
            "Linear Regression": (LinearRegression(), "scaled"),
        }

        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = (
                XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.06,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    objective="reg:squarederror",
                ),
                "raw",
            )
        except Exception:
            pass

        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = (
                LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.06,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                ),
                "raw",
            )
        except Exception:
            pass

        return models

    def train(
        self, file_path: str = "data/Student Attitude and Behavior.csv", data_dir: str = "data"
    ) -> Any:
        pipeline = prepare_training_dataset(base_dataset_path=file_path, data_dir=data_dir)
        df = pipeline.data

        self.dataset_counts = pipeline.dataset_counts
        self.used_files = pipeline.used_files
        features = df[MODEL_FEATURES].copy()
        y = df[TARGET_COLUMN].astype(float).values
        self.feature_columns = MODEL_FEATURES.copy()

        features_train, features_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=0.25,
            random_state=42,
        )
        self.scaler.fit(features_train)
        features_train_scaled = pd.DataFrame(
            self.scaler.transform(features_train), columns=self.feature_columns
        )
        features_test_scaled = pd.DataFrame(
            self.scaler.transform(features_test), columns=self.feature_columns
        )

        best_model = None
        best_name = None
        best_mode = None
        best_mae = float("inf")

        for model_name, (model, mode) in self._build_candidate_models().items():
            train_matrix = features_train_scaled if mode == "scaled" else features_train
            test_matrix = features_test_scaled if mode == "scaled" else features_test

            model.fit(train_matrix, y_train)
            pred = model.predict(test_matrix)
            mae = float(mean_absolute_error(y_test, pred))

            cv_scores = cross_val_score(
                model,
                train_matrix,
                y_train,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            self.cv_results[model_name] = {
                "cv_mae": float(np.abs(cv_scores.mean())),
                "cv_std": float(cv_scores.std()),
                "test_mae": mae,
            }

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = model_name
                best_mode = mode

        self.model = best_model
        self.model_name = best_name
        self.model_type = best_mode or "raw"

        self.training_reference = features_train.head(min(120, len(features_train))).reset_index(
            drop=True
        )
        if self.model_type == "scaled":
            self.training_reference = pd.DataFrame(
                self.scaler.transform(self.training_reference), columns=self.feature_columns
            )

        final_train_matrix = (
            features_train_scaled if self.model_type == "scaled" else features_train
        )
        selected_model = self._ensure_model()
        train_pred = selected_model.predict(final_train_matrix)
        residuals = y_train - train_pred
        self.residual_std = float(max(np.std(residuals), 5.0))
        return self.model

    def _ensure_model(self) -> Any:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model

    def _predict_raw_score(self, prepared_df: pd.DataFrame) -> NDArray[np.float64]:
        model = self._ensure_model()
        matrix = prepared_df[self.feature_columns]
        if self.model_type == "scaled":
            matrix = pd.DataFrame(self.scaler.transform(matrix), columns=self.feature_columns)
        pred = np.asarray(model.predict(matrix), dtype=np.float64)
        return np.clip(pred, 0.0, 100.0)

    def predict(self, input_data: dict[str, Any] | pd.DataFrame) -> float:
        profile = self.predict_profile(input_data)
        return float(profile["readiness_score"])

    def _domain_scores_from_row(self, row: pd.Series) -> dict[str, float]:
        scores = {
            "Web Dev": float(row.get("domain_web_dev", 0.0)),
            "Data": float(row.get("domain_data", 0.0)),
            "DevOps": float(row.get("domain_devops", 0.0)),
            "Mobile": float(row.get("domain_mobile", 0.0)),
        }
        return {k: float(np.clip(v, 0.0, 100.0)) for k, v in scores.items()}

    def predict_profile(self, input_data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        frame = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()

        prepared = normalize_input_frame(frame)
        readiness = float(self._predict_raw_score(prepared)[0])
        ci_margin = 1.96 * self.residual_std
        confidence_low = float(np.clip(readiness - ci_margin, 0.0, 100.0))
        confidence_high = float(np.clip(readiness + ci_margin, 0.0, 100.0))

        domain_scores = self._domain_scores_from_row(prepared.iloc[0])
        best_domain = max(domain_scores, key=lambda name: domain_scores[name])
        readiness_level = "High" if readiness >= 70 else "Moderate" if readiness >= 40 else "Low"

        return {
            "readiness_score": readiness,
            "readiness_level": readiness_level,
            "confidence_interval": [confidence_low, confidence_high],
            "best_domain": best_domain,
            "domain_scores": domain_scores,
            "feature_values": prepared.iloc[0][self.feature_columns].to_dict(),
        }

    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        prepared = normalize_input_frame(input_data)
        readiness_scores = self._predict_raw_score(prepared)

        ci_margin = 1.96 * self.residual_std
        result = input_data.copy()
        result["predicted_junior_readiness"] = np.clip(readiness_scores, 0.0, 100.0)
        result["confidence_low"] = np.clip(
            result["predicted_junior_readiness"] - ci_margin, 0.0, 100.0
        )
        result["confidence_high"] = np.clip(
            result["predicted_junior_readiness"] + ci_margin, 0.0, 100.0
        )
        result["readiness_level"] = pd.cut(
            result["predicted_junior_readiness"],
            bins=[-0.01, 40, 70, 100],
            labels=["Low", "Moderate", "High"],
        ).astype(str)

        for col in DOMAIN_COLUMNS:
            if col in prepared.columns:
                result[col] = prepared[col]
            else:
                result[col] = 0.0

        domain_map = {
            "domain_web_dev": "Web Dev",
            "domain_data": "Data",
            "domain_devops": "DevOps",
            "domain_mobile": "Mobile",
        }
        result["best_domain"] = result[list(domain_map.keys())].idxmax(axis=1).map(domain_map)
        return result

    def get_feature_importance(self) -> pd.DataFrame | None:
        model = self._ensure_model()

        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=np.float64)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_, dtype=np.float64).flatten())
        else:
            return None

        return pd.DataFrame({"feature": self.feature_columns, "importance": values}).sort_values(
            "importance", ascending=False
        )

    def get_shap_feature_importance(
        self, input_frame: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        self._ensure_model()
        if self.training_reference is None:
            return None

        try:
            import shap
        except Exception:
            return None

        model_type_name = type(self.model).__name__
        if model_type_name in ["SVR", "LinearRegression"]:
            return None

        if input_frame is None:
            matrix = self.training_reference.copy()
        else:
            frame = input_frame.copy()
            if set(self.feature_columns).issubset(frame.columns):
                matrix = frame[self.feature_columns]
            else:
                matrix = self.training_reference.copy()
            if self.model_type == "scaled":
                matrix = pd.DataFrame(self.scaler.transform(matrix), columns=self.feature_columns)

        try:
            explainer = shap.Explainer(self.model, self.training_reference)
            shap_values = explainer(matrix)
            values = np.abs(np.asarray(shap_values.values)).mean(axis=0)
        except Exception:
            return None

        return pd.DataFrame({"feature": self.feature_columns, "importance": values}).sort_values(
            "importance", ascending=False
        )

    def skill_gap_analysis(
        self, input_data: dict[str, Any] | pd.DataFrame, target_role: str
    ) -> dict[str, Any]:
        frame = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
        prepared = normalize_input_frame(frame).iloc[0]

        role_thresholds: dict[str, dict[str, float]] = {
            "Web Dev": {
                "github_activity_proxy": 5.0,
                "side_projects_count": 3.0,
                "languages_known_count": 3.0,
                "tech_skills_self_rating": 6.0,
            },
            "Data": {
                "college_mark": 75.0,
                "daily_studying_time": 3.0,
                "tech_skills_self_rating": 6.5,
                "certification_course": 1.0,
            },
            "DevOps": {
                "internship_experience": 1.0,
                "github_activity_proxy": 6.0,
                "side_projects_count": 2.0,
                "stress_level": 2.0,
            },
            "Mobile": {
                "side_projects_count": 3.0,
                "languages_known_count": 3.0,
                "tech_skills_self_rating": 6.0,
                "likes_degree": 1.0,
            },
        }

        thresholds = role_thresholds.get(target_role, role_thresholds["Web Dev"])
        missing: list[str] = []
        met: list[str] = []
        labels = {
            "github_activity_proxy": "GitHub activity",
            "side_projects_count": "side projects",
            "languages_known_count": "languages known",
            "tech_skills_self_rating": "tech skills self-rating",
            "college_mark": "college marks",
            "daily_studying_time": "daily studying time",
            "certification_course": "certification",
            "internship_experience": "internship/co-op experience",
            "stress_level": "stress management",
            "likes_degree": "degree alignment",
        }

        for feature, minimum in thresholds.items():
            value = float(prepared.get(feature, DEFAULT_FEATURE_VALUES.get(feature, 0.0)))
            is_met = value <= minimum if feature == "stress_level" else value >= minimum
            if is_met:
                met.append(labels.get(feature, feature))
            else:
                missing.append(labels.get(feature, feature))

        return {"target_role": target_role, "met": met, "missing": missing}

    def save_model(self, filepath: str = "student_career_willingness_model.pkl") -> None:
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "residual_std": self.residual_std,
            "cv_results": self.cv_results,
            "dataset_counts": self.dataset_counts,
            "used_files": self.used_files,
            "training_reference": self.training_reference,
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str = "student_career_willingness_model.pkl") -> None:
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.model_name = model_data.get("model_name")
        self.model_type = model_data.get("model_type", "raw")
        self.residual_std = model_data.get("residual_std", 8.0)
        self.cv_results = model_data.get("cv_results", {})
        self.dataset_counts = model_data.get("dataset_counts", {})
        self.used_files = model_data.get("used_files", [])
        self.training_reference = model_data.get("training_reference")
