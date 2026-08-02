from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .data_pipeline import MODEL_FEATURES, TARGET_COLUMN, prepare_training_dataset
from .student_career_willingness_predictor import StudentCareerWillingnessPredictor

MODEL_REGISTRY_NAME = "student-readiness-model"


def _feature_importance(
    model: Any,
    features: pd.DataFrame,
    target: np.ndarray[Any, Any],
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=np.float64).ravel())
    else:
        values = permutation_importance(
            model,
            features,
            target,
            n_repeats=5,
            random_state=42,
        ).importances_mean
    return pd.DataFrame({"feature": MODEL_FEATURES, "importance": values}).sort_values(
        "importance", ascending=False
    )


def _save_shap_summary(model: Any, features: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    try:
        import shap

        background = shap.sample(features, min(40, len(features)), random_state=42)
        sample = features.head(min(20, len(features)))
        explainer = shap.Explainer(model.predict, background)
        values = explainer(sample)
        shap.summary_plot(values, sample, show=False)
    except Exception as exc:
        plt.text(0.5, 0.5, f"SHAP summary unavailable: {exc}", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def train_with_mlflow(
    data_dir: str = "data",
    model_output_path: str = "models/student_career_willingness_model.pkl",
    experiment_name: str = "student-career-readiness",
) -> StudentCareerWillingnessPredictor:
    mlflow.set_experiment(experiment_name)
    pipeline = prepare_training_dataset(data_dir=data_dir)
    data = pipeline.data
    features = data[MODEL_FEATURES].copy()
    target = data[TARGET_COLUMN].astype(float).to_numpy()
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    scaler = StandardScaler().fit(features_train)
    features_train_scaled = pd.DataFrame(scaler.transform(features_train), columns=MODEL_FEATURES)
    features_test_scaled = pd.DataFrame(scaler.transform(features_test), columns=MODEL_FEATURES)

    predictor = StudentCareerWillingnessPredictor(load_on_init=False)
    predictor.scaler = scaler
    predictor.dataset_counts = pipeline.dataset_counts
    predictor.used_files = pipeline.used_files

    best_cv_mae = float("inf")
    best_model: Any = None
    best_model_name = ""
    best_mode = "raw"
    best_run_id = ""

    for model_name, (model, mode) in predictor._build_candidate_models().items():
        train_matrix = features_train_scaled if mode == "scaled" else features_train
        test_matrix = features_test_scaled if mode == "scaled" else features_test
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_params({key: str(value) for key, value in model.get_params().items()})
            model.fit(train_matrix, target_train)
            predictions = model.predict(test_matrix)
            cv_scores = cross_val_score(
                model,
                train_matrix,
                target_train,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            cv_mae = float(np.abs(cv_scores.mean()))
            test_mae = float(mean_absolute_error(target_test, predictions))
            test_rmse = float(np.sqrt(mean_squared_error(target_test, predictions)))
            mlflow.log_metrics({"cv_mae": cv_mae, "test_mae": test_mae, "test_rmse": test_rmse})

            reference = train_matrix.head(min(120, len(train_matrix))).reset_index(drop=True)
            bundle = {
                "model": model,
                "scaler": scaler,
                "feature_columns": MODEL_FEATURES,
                "model_name": model_name,
                "model_type": mode,
                "residual_std": float(max(np.std(target_train - model.predict(train_matrix)), 5.0)),
                "cv_results": {
                    model_name: {
                        "cv_mae": cv_mae,
                        "cv_std": float(cv_scores.std()),
                        "test_mae": test_mae,
                    }
                },
                "dataset_counts": pipeline.dataset_counts,
                "used_files": pipeline.used_files,
                "training_reference": reference,
            }
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = Path(temp_dir)
                joblib.dump(model, artifact_dir / "fitted_model.pkl")
                joblib.dump(bundle, artifact_dir / "predictor_bundle.pkl")
                importance = _feature_importance(model, test_matrix, target_test)
                importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
                _save_shap_summary(model, test_matrix, artifact_dir / "shap_summary.png")
                mlflow.log_artifacts(str(artifact_dir))
            mlflow.sklearn.log_model(model, name="model")

            predictor.cv_results[model_name] = bundle["cv_results"][model_name]
            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_model = model
                best_model_name = model_name
                best_mode = mode
                best_run_id = run.info.run_id

    predictor.model = best_model
    predictor.model_name = best_model_name
    predictor.model_type = best_mode
    best_train_matrix = features_train_scaled if best_mode == "scaled" else features_train
    predictor.training_reference = best_train_matrix.head(
        min(120, len(best_train_matrix))
    ).reset_index(drop=True)
    predictor.residual_std = float(
        max(np.std(target_train - best_model.predict(best_train_matrix)), 5.0)
    )
    predictor.save_model(model_output_path)

    registered = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_REGISTRY_NAME)
    MlflowClient().set_registered_model_alias(MODEL_REGISTRY_NAME, "latest", registered.version)
    return predictor


if __name__ == "__main__":
    train_with_mlflow()
