"""
Tests for model_building. Models are kept tiny (no grid search, 2-fold CV, 1 repeat) so each runs in seconds.

These check KIF's wiring around the ML libraries, not how well the models learn (nothing is seeded).
"""

import numpy as np
import pytest

from key_interactions_finder.model_building import ClassificationModel, RegressionModel
from tests.conftest import SYNTH_FEATURES

SEPARATOR = SYNTH_FEATURES[0]
FAST = {"cross_validation_splits": 2, "cross_validation_repeats": 1, "search_approach": "none"}


def _classification_model(dataset, models, **kwargs):
    return ClassificationModel(dataset=dataset, models_to_use=models, classes_to_use=[], **FAST, **kwargs)


def test_classification_setup(in_tmp_dir, synthetic_supervised_df):
    model = _classification_model(synthetic_supervised_df, ["Random_Forest"], evaluation_split_ratio=0.25)
    assert list(model.feat_names) == SYNTH_FEATURES
    assert len(model.ml_datasets["train_data_scaled"]) == 45
    assert len(model.ml_datasets["eval_data_scaled"]) == 15
    # Classes are label encoded (XGBoost needs 0..n-1).
    assert set(np.unique(model.ml_datasets["y_train"])) == {0, 1}
    assert list(model.label_encoder.classes_) == ["A", "B"]
    # min_max scaling is fitted on the training data only.
    assert model.ml_datasets["train_data_scaled"].min() == pytest.approx(0.0)
    assert model.ml_datasets["train_data_scaled"].max() == pytest.approx(1.0)


def test_classification_class_weights(in_tmp_dir, synthetic_supervised_df):
    dataset = synthetic_supervised_df.iloc[:40]  # 30 "A", 10 "B".
    model = _classification_model(dataset, ["XGBoost"], use_class_weights=True)
    y_train = model.ml_datasets["y_train"]
    expected = round(np.sum(y_train == 0) / np.sum(y_train == 1), 2)
    assert model.available_models["XGBoost"]["model"].get_params()["scale_pos_weight"] == expected


def test_classification_classes_to_use(in_tmp_dir, synthetic_supervised_df):
    dataset = synthetic_supervised_df.copy()
    dataset.loc[:4, "Target"] = "C"
    model = ClassificationModel(dataset=dataset, models_to_use=["Random_Forest"], classes_to_use=["A", "B"], **FAST)
    assert "C" not in set(model.dataset["Target"])
    assert len(model.dataset) == 55


@pytest.mark.parametrize("models", [["Random_Forest"], ["XGBoost"], ["CatBoost"]])
def test_classification_build_evaluate(in_tmp_dir, synthetic_supervised_df, models):
    model = _classification_model(synthetic_supervised_df, models)
    results = model.build_models(save_models=True)

    assert results["model"].tolist() == models

    reports = model.evaluate_models()
    assert set(reports[models[0]].index) == {"A", "B", "accuracy", "macro avg", "weighted avg"}

    matrices = model.generate_confusion_matrix()
    assert matrices[models[0]].shape == (2, 2)

    # build_models always writes the models to ./temporary_files.
    assert (in_tmp_dir / "temporary_files" / f"{models[0]}_Model.pickle").exists()
    assert (in_tmp_dir / "temporary_files" / "feature_names.npy").exists()


@pytest.mark.parametrize("models", [["Random_Forest"], ["XGBoost"], ["CatBoost"]])
def test_regression_build_evaluate(in_tmp_dir, synthetic_regression_df, models):
    model = RegressionModel(dataset=synthetic_regression_df, models_to_use=models, **FAST)
    results = model.build_models(save_models=False)
    assert results["model"].tolist() == models
    assert not (in_tmp_dir / "temporary_files").exists()

    metrics = model.evaluate_models()
    assert metrics["Model"].tolist() == models


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    df = RegressionModel._regression_metrics("test", y_true, y_pred)
    assert df.loc[0, "Model"] == "test"
    assert df.loc[0, "Mean Absolute Error"] == 0.25
    assert df.loc[0, "MSE"] == 0.25
    assert df.loc[0, "RMSE"] == 0.5
    assert df.loc[0, "r squared"] == 0.8


def test_regression_metrics_negative_targets():
    df = RegressionModel._regression_metrics("test", np.array([-1.0, 2.0]), np.array([-1.0, 2.0]))
    assert df.loc[0, "Mean Squared Log Error"] == "N/A"
