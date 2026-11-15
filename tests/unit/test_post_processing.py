"""Tests for post_proccessing: per feature -> per residue projection and the post-processor classes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from key_interactions_finder.model_building import ClassificationModel
from key_interactions_finder.post_proccessing import (
    PostProcessor,
    StatClassificationPostProcessor,
    StatRegressorPostProcessor,
    SupervisedPostProcessor,
)
from key_interactions_finder.stat_modelling import ClassificationStatModel, RegressionStatModel
from tests.conftest import SYNTH_FEATURES

ALL_TYPES = ["Hbond", "Hydrophobic", "Saltbr", "Other"]


def _per_res(feature_scores: dict) -> dict:
    return PostProcessor._per_res_scores(PostProcessor._dict_to_df_feat_scores(feature_scores))


def test_dict_to_df_feat_scores():
    df = PostProcessor._dict_to_df_feat_scores({"5Lys 9Glu Saltbr": 0.7, "12Ala 100Leu Hydrophobic bb-sc": -0.3})
    assert df["Res1"].tolist() == [5, 12]
    assert df["Res2"].tolist() == [9, 100]
    # Absolute values so negative correlations count towards residue scores.
    assert df["Score"].tolist() == [0.7, 0.3]


def test_per_res_scores_on_correct_residues():
    """Scores land on the residues named in the feature, not their neighbours."""
    scores = _per_res({"5Lys 9Glu Saltbr": 0.7})
    assert scores[5] == 1.0
    assert scores[9] == 1.0
    assert scores.get(6, 0) == 0
    assert scores.get(10, 0) == 0


def test_per_res_scores_sums_and_normalises():
    scores = _per_res({"1Ala 5Leu Hydrophobic": 0.2, "5Leu 20Glu Other": 0.6, "20Glu 30Lys Saltbr": 0.2})
    # Residue 5 = 0.2 + 0.6, residue 20 = 0.6 + 0.2: both are the max (0.8) -> 1.0.
    assert scores[5] == 1.0
    assert scores[20] == 1.0
    assert scores[1] == 0.25
    assert scores[30] == 0.25
    # Ordered by score, highest first.
    values = list(scores.values())
    assert values == sorted(values, reverse=True)


def test_per_res_scores_residue_one_included():
    scores = _per_res({"1Ala 5Leu Hydrophobic": 1.0})
    assert scores[1] == 1.0


def test_per_res_scores_include_all_residues_up_to_max():
    scores = _per_res({"3Ala 7Leu Hydrophobic": 1.0})
    assert sorted(scores) == list(range(1, 8))


def test_per_res_and_feature_files(tmp_path):
    PostProcessor._per_feature_scores_to_file({"1Ala 5Leu Hydrophobic": 0.123456}, tmp_path / "feat.csv")
    PostProcessor._per_res_scores_to_file({5: 1.0, 1: 0.5}, tmp_path / "res.csv")
    assert (tmp_path / "feat.csv").read_text().splitlines() == ["Feature,Score", "1Ala 5Leu Hydrophobic,0.1235"]
    assert (tmp_path / "res.csv").read_text().splitlines() == ["Residue Number,Normalised Score", "5,1.0", "1,0.5"]


@pytest.fixture
def stat_class_post(synthetic_supervised_df, tmp_path):
    model = ClassificationStatModel(
        dataset=synthetic_supervised_df,
        class_names=["A", "B"],
        out_dir=str(tmp_path),
        interaction_types_included=ALL_TYPES,
    )
    model.calc_js_distances(save_result=False)
    model.calc_mutual_info_to_target(save_result=False)
    return StatClassificationPostProcessor(stat_model=model, out_dir=str(tmp_path))


@pytest.mark.parametrize(
    ("method", "out_file"),
    [
        ("jensen_shannon", "Jensen_Shannon_Distance_Scores_Per_Residue.csv"),
        ("mutual_information", "Mutual_Information_Scores_Per_Residue.csv"),
    ],
)
def test_stat_class_per_res_scores(stat_class_post, tmp_path, method, out_file):
    scores = stat_class_post.get_per_res_scores(stat_method=method)
    assert max(scores.values()) == 1.0
    # The separating feature is "1Ala 5Leu": residues 1 and 5 should be near the top.
    assert {1, 5} & set(list(scores)[:3])
    saved = pd.read_csv(tmp_path / out_file)
    assert saved["Residue Number"].tolist() == list(scores)


def test_stat_class_per_res_bad_method(stat_class_post):
    with pytest.raises(ValueError):
        stat_class_post.get_per_res_scores(stat_method="nonsense")


def test_stat_class_get_kdes(stat_class_post):
    x_values, kdes = stat_class_post.get_kdes(number_features=2)
    assert x_values.shape == (50, 1)
    assert list(kdes["A"]) == list(stat_class_post.stat_model.js_distances)[:2]

    _, all_kdes = stat_class_post.get_kdes(number_features="all")
    assert set(all_kdes["A"]) == set(SYNTH_FEATURES)


def test_stat_class_feature_directions(stat_class_post, tmp_path):
    with pytest.warns(UserWarning, match="very simplistic"):
        stat_class_post.estimate_feature_directions()
    # The separator is low in class A and high in class B.
    assert stat_class_post.feature_directions[SYNTH_FEATURES[0]] == "B"
    assert (tmp_path / "Feature_Direction_Estimates.csv").exists()


@pytest.mark.parametrize(
    ("method", "out_file"),
    [
        ("linear_correlation", "Linear_Correlation_Scores_Per_Residue.csv"),
        ("mutual_information", "Mutual_Information_Scores_Per_Residue.csv"),
    ],
)
def test_stat_regression_per_res_scores(synthetic_regression_df, tmp_path, method, out_file):
    model = RegressionStatModel(
        dataset=synthetic_regression_df, out_dir=str(tmp_path), interaction_types_included=ALL_TYPES
    )
    model.calc_linear_correl_to_target(save_result=False)
    model.calc_mutual_info_to_target(save_result=False)
    post = StatRegressorPostProcessor(stat_model=model, out_dir=str(tmp_path))

    scores = post.get_per_res_scores(stat_method=method)
    assert set(list(scores)[:2]) == {1, 5}
    assert (tmp_path / out_file).exists()

    with pytest.raises(ValueError):
        post.get_per_res_scores(stat_method="jensen_shannon")


@pytest.fixture
def built_rf_model(in_tmp_dir, synthetic_supervised_df):
    model = ClassificationModel(
        dataset=synthetic_supervised_df,
        models_to_use=["Random_Forest"],
        classes_to_use=[],
        cross_validation_splits=2,
        cross_validation_repeats=1,
    )
    model.build_models(save_models=True)
    return model


def test_supervised_post_from_instance(built_rf_model, in_tmp_dir):
    post = SupervisedPostProcessor(out_dir="results")
    post.load_models_from_instance(supervised_model=built_rf_model)
    post.get_per_feature_scores()
    post.get_per_res_scores()

    assert set(post.all_per_feature_scores["Random_Forest"]) == set(SYNTH_FEATURES)
    assert (in_tmp_dir / "results" / "Random_Forest_Feature_Scores.csv").exists()
    assert (in_tmp_dir / "results" / "Random_Forest_Per_Residue_Scores.csv").exists()


def test_supervised_post_maps_importances_to_residues(tmp_path):
    """Stub model with fixed importances, so KIF's importance -> feature -> residue mapping is checked exactly."""
    post = SupervisedPostProcessor(out_dir=str(tmp_path))
    post.feat_names = np.array(["1Ala 5Leu Hydrophobic", "5Leu 20Glu Other", "20Glu 30Lys Saltbr"])
    post.best_models = {"Stub": SimpleNamespace(feature_importances_=np.array([0.2, 0.6, 0.2]))}
    post.get_per_res_scores()  # also runs get_per_feature_scores.

    assert post.all_per_feature_scores["Stub"] == {
        "5Leu 20Glu Other": 0.6,
        "1Ala 5Leu Hydrophobic": 0.2,
        "20Glu 30Lys Saltbr": 0.2,
    }
    per_res = post.all_per_residue_scores["Stub"]
    assert (per_res[5], per_res[20], per_res[1], per_res[30]) == (1.0, 1.0, 0.25, 0.25)
    assert (tmp_path / "Stub_Per_Residue_Scores.csv").exists()


def test_supervised_post_from_disk_matches_instance(built_rf_model):
    from_instance = SupervisedPostProcessor()
    from_instance.load_models_from_instance(supervised_model=built_rf_model, models_to_use=["Random_Forest"])
    from_instance.get_per_feature_scores(save_result=False)

    from_disk = SupervisedPostProcessor()
    from_disk.load_models_from_disk(models_to_use=["Random_Forest"])
    from_disk.get_per_feature_scores(save_result=False)

    assert from_disk.all_per_feature_scores == from_instance.all_per_feature_scores


def test_supervised_post_bad_models_to_use(built_rf_model):
    with pytest.raises(ValueError):
        SupervisedPostProcessor().load_models_from_instance(supervised_model=built_rf_model, models_to_use="RF")


def test_supervised_post_from_disk_missing(in_tmp_dir):
    with pytest.raises(FileNotFoundError, match="temporary_files"):
        SupervisedPostProcessor().load_models_from_disk(models_to_use=["Random_Forest"])
