"""Tests for stat_modelling: ClassificationStatModel and RegressionStatModel."""

import numpy as np
import pandas as pd
import pytest

from key_interactions_finder.stat_modelling import ClassificationStatModel, RegressionStatModel
from tests.conftest import SYNTH_FEATURES

ALL_TYPES = ["Hbond", "Hydrophobic", "Saltbr", "Other"]
SEPARATOR = SYNTH_FEATURES[0]


@pytest.fixture
def class_model(synthetic_supervised_df, tmp_path):
    return ClassificationStatModel(
        dataset=synthetic_supervised_df,
        class_names=["A", "B"],
        out_dir=str(tmp_path),
        interaction_types_included=ALL_TYPES,
    )


@pytest.fixture
def reg_model(synthetic_regression_df, tmp_path):
    return RegressionStatModel(
        dataset=synthetic_regression_df, out_dir=str(tmp_path), interaction_types_included=ALL_TYPES
    )


def test_scaled_dataset_between_0_and_1(class_model):
    features = class_model.scaled_dataset.drop("Target", axis=1)
    assert features.min().min() == pytest.approx(0.0)
    assert features.max().max() == pytest.approx(1.0)
    assert class_model.feature_list == SYNTH_FEATURES


def test_scaled_dataset_target_aligned_with_non_default_index(synthetic_supervised_df, tmp_path):
    """Targets stay with their rows when the index is not 0..n-1, e.g. after dropping rows."""
    dataset = synthetic_supervised_df.drop(index=[0, 1, 2, 40, 41])
    model = ClassificationStatModel(
        dataset=dataset, class_names=["A", "B"], out_dir=str(tmp_path), interaction_types_included=ALL_TYPES
    )
    assert model.scaled_dataset["Target"].isna().sum() == 0
    assert model.scaled_dataset["Target"].tolist() == dataset["Target"].tolist()


def test_interaction_type_filtering(synthetic_supervised_df, tmp_path):
    model = ClassificationStatModel(
        dataset=synthetic_supervised_df,
        class_names=["A", "B"],
        out_dir=str(tmp_path),
        interaction_types_included=["Hydrophobic"],
    )
    assert model.feature_list == [SYNTH_FEATURES[0], SYNTH_FEATURES[4]]


def test_requires_exactly_two_classes(synthetic_supervised_df, tmp_path):
    with pytest.raises(ValueError, match="should be 2"):
        ClassificationStatModel(
            dataset=synthetic_supervised_df,
            class_names=["A"],
            out_dir=str(tmp_path),
            interaction_types_included=ALL_TYPES,
        )


def test_js_distances_rank_separator_first(class_model, tmp_path):
    class_model.calc_js_distances()
    assert list(class_model.js_distances)[0] == SEPARATOR
    # Non-overlapping distributions -> JS distance (base 2) close to the maximum of 1.
    assert class_model.js_distances[SEPARATOR] > 0.95
    assert all(0 <= v <= 1 for v in class_model.js_distances.values())
    # Sorted, highest first.
    values = list(class_model.js_distances.values())
    assert values == sorted(values, reverse=True)

    saved = pd.read_csv(tmp_path / "Jensen_Shannon_Per_Feature_Scores.csv")
    assert saved["Feature"].tolist() == list(class_model.js_distances)


def test_kdes_shape(class_model):
    class_model.calc_js_distances(kde_bandwidth=0.02, save_result=False)
    assert class_model.x_values.shape == (50, 1)
    assert set(class_model.kdes) == {"A", "B"}
    assert set(class_model.kdes["A"]) == set(SYNTH_FEATURES)
    assert class_model.kdes["A"][SEPARATOR].shape == (50,)


def test_mutual_info_ranks_separator_first(class_model, tmp_path):
    class_model.calc_mutual_info_to_target()
    assert list(class_model.mutual_infos)[0] == SEPARATOR
    assert (tmp_path / "Mutual_Information_Per_Feature_Scores.csv").exists()


def test_save_result_false_writes_nothing(class_model, tmp_path):
    class_model.calc_js_distances(save_result=False)
    class_model.calc_mutual_info_to_target(save_result=False)
    assert list(tmp_path.iterdir()) == []


def test_regression_linear_correlation(reg_model, synthetic_regression_df, tmp_path):
    reg_model.calc_linear_correl_to_target()
    assert list(reg_model.linear_correlations)[0] == SEPARATOR
    assert reg_model.linear_correlations[SEPARATOR] > 0.99
    expected = synthetic_regression_df[SEPARATOR].corr(synthetic_regression_df["Target"])
    assert reg_model.linear_correlations[SEPARATOR] == pytest.approx(expected)
    assert (tmp_path / "Linear_Correlations_Per_Feature_Scores.csv").exists()


def test_regression_mutual_info(reg_model):
    reg_model.calc_mutual_info_to_target(save_result=False)
    assert list(reg_model.mutual_infos)[0] == SEPARATOR


def test_regression_target_aligned_with_non_default_index(synthetic_regression_df, tmp_path):
    dataset = synthetic_regression_df.drop(index=[0, 5, 10])
    model = RegressionStatModel(dataset=dataset, out_dir=str(tmp_path), interaction_types_included=ALL_TYPES)
    np.testing.assert_allclose(model.scaled_dataset["Target"].to_numpy(), dataset["Target"].to_numpy())
