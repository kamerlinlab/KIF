"""Tests for data_preperation: loading targets and the filter chain."""

import pandas as pd
import pytest

from key_interactions_finder.data_preperation import SupervisedFeatureData, UnsupervisedFeatureData
from tests.conftest import SYNTH_FEATURES

SEPARATOR, LOW_OCC, WEAK, NOISE, NOISE_11 = SYNTH_FEATURES


@pytest.fixture
def supervised(synthetic_features, synthetic_target_file):
    return SupervisedFeatureData(input_df=synthetic_features, is_classification=True, target_file=synthetic_target_file)


@pytest.fixture
def unsupervised(synthetic_features):
    return UnsupervisedFeatureData(input_df=synthetic_features)


def test_supervised_merges_target_first(supervised, synthetic_class_labels):
    assert list(supervised.df_processed.columns) == ["Target", *SYNTH_FEATURES]
    assert supervised.df_processed["Target"].tolist() == synthetic_class_labels
    assert supervised.df_filtered.empty


def test_supervised_target_without_header(synthetic_features, synthetic_class_labels, tmp_path):
    target_file = tmp_path / "classes_no_header.txt"
    target_file.write_text("\n".join(synthetic_class_labels) + "\n")
    data = SupervisedFeatureData(
        input_df=synthetic_features, is_classification=True, target_file=str(target_file), header_present=False
    )
    assert data.df_processed["Target"].tolist() == synthetic_class_labels


def test_supervised_length_mismatch_raises(synthetic_features, synthetic_target_file):
    with pytest.raises(ValueError, match="doesn't match"):
        SupervisedFeatureData(
            input_df=synthetic_features.iloc[:-1], is_classification=True, target_file=synthetic_target_file
        )


def test_unsupervised_processed_is_input(unsupervised, synthetic_features):
    pd.testing.assert_frame_equal(unsupervised.df_processed, synthetic_features)


def test_filter_by_occupancy(supervised):
    result = supervised.filter_by_occupancy(min_occupancy=50)
    assert list(result.columns) == ["Target", SEPARATOR, WEAK, NOISE, NOISE_11]
    assert result is supervised.df_filtered


def test_filter_by_occupancy_keeps_mostly_zero_target(synthetic_features, tmp_path):
    """A 0/1 target that is mostly 0 must not be removed by the occupancy filter."""
    target_file = tmp_path / "binary.csv"
    pd.DataFrame({"Target": [0] * 50 + [1] * 10}).to_csv(target_file, index=False)
    data = SupervisedFeatureData(input_df=synthetic_features, is_classification=True, target_file=str(target_file))
    assert "Target" in data.filter_by_occupancy(min_occupancy=50).columns


def test_filter_by_avg_strength(supervised):
    result = supervised.filter_by_avg_strength(average_strength_cut_off=0.05)
    assert WEAK not in result.columns
    assert result.columns[0] == "Target"


def test_filter_by_interaction_type(supervised):
    result = supervised.filter_by_interaction_type(["Hbond", "Other"])
    assert list(result.columns) == ["Target", LOW_OCC, NOISE]


def test_filter_by_occupancy_by_class(synthetic_features, synthetic_target_file):
    # Make the low occupancy feature fully occupied in class "A" only.
    df = synthetic_features.copy()
    df[LOW_OCC] = [1.0] * 30 + [0.0] * 30
    data = SupervisedFeatureData(input_df=df, is_classification=True, target_file=synthetic_target_file)

    # Overall occupancy is 50%, so a 75% overall cut-off removes it...
    assert LOW_OCC not in data.filter_by_occupancy(min_occupancy=75).columns
    data.reset_filtering()
    # ...but the per class cut-off keeps it, as class "A" alone meets it.
    result = data.filter_by_occupancy_by_class(min_occupancy=75)
    assert LOW_OCC in result.columns
    assert result.columns[0] == "Target"


def test_filter_by_occupancy_by_class_rejects_regression(synthetic_features, synthetic_target_file):
    data = SupervisedFeatureData(
        input_df=synthetic_features, is_classification=False, target_file=synthetic_target_file
    )
    with pytest.raises(TypeError):
        data.filter_by_occupancy_by_class(min_occupancy=50)


def test_filter_by_main_or_side_chain(synthetic_features, synthetic_target_file):
    df = synthetic_features.copy()
    df.columns = [
        f"{col} {chain}" for col, chain in zip(df.columns, ["sc-sc", "bb-bb", "sc-bb", "bb-sc", "sc-sc"], strict=True)
    ]
    data = SupervisedFeatureData(input_df=df, is_classification=True, target_file=synthetic_target_file)
    result = data.filter_by_main_or_side_chain(["sc-sc"])
    assert list(result.columns) == ["Target", f"{SEPARATOR} sc-sc", f"{NOISE_11} sc-sc"]


def test_filter_by_main_or_side_chain_needs_chain_info(supervised):
    with pytest.raises(ValueError, match="side or main chain"):
        supervised.filter_by_main_or_side_chain(["sc-sc"])


def test_reset_filtering(supervised):
    supervised.filter_by_occupancy(min_occupancy=50)
    result = supervised.reset_filtering()
    pd.testing.assert_frame_equal(result, supervised.df_processed)
