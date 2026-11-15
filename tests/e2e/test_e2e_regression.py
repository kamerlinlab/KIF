"""
End-to-end: contacts CSV with a continuous target -> regression stats + ML -> per residue scores -> PyMOL files.

Follows tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb. The KE07 dataset is not committed, so this uses the
PTP1B subset with a synthetic continuous target: the Asp180-Arg220 salt bridge strength (removed from the features)
plus noise. Features from the WPD-loop / Arg220, which move with that salt bridge, should then score highest.
"""

import re

import numpy as np
import pandas as pd
import pytest

from key_interactions_finder import pymol_projections
from key_interactions_finder.data_preperation import SupervisedFeatureData
from key_interactions_finder.model_building import RegressionModel
from key_interactions_finder.post_proccessing import StatRegressorPostProcessor, SupervisedPostProcessor
from key_interactions_finder.stat_modelling import RegressionStatModel
from tests.conftest import PTP1B_CONTACTS

TARGET_FEATURE = "180Asp 220Arg Saltbr"
# Residues expected to move with the target: the WPD-loop and Arg220.
KEY_RESIDUES = set(range(176, 190)) | {220}


@pytest.fixture(scope="module")
def regression_dataset(tmp_path_factory) -> pd.DataFrame:
    tmp_dir = tmp_path_factory.mktemp("ptp1b_regression_input")
    contacts = pd.read_csv(PTP1B_CONTACTS)

    rng = np.random.default_rng(0)
    target = contacts[TARGET_FEATURE] + rng.normal(0, 0.5, len(contacts))
    target_file = tmp_dir / "target.csv"
    pd.DataFrame({"SaltBridge": target}).to_csv(target_file, index=False)

    dataset = SupervisedFeatureData(
        input_df=contacts.drop(columns=[TARGET_FEATURE]),
        target_file=str(target_file),
        is_classification=False,
        header_present=True,
    )
    dataset.reset_filtering()
    dataset.filter_by_occupancy(min_occupancy=25)
    dataset.filter_by_avg_strength(average_strength_cut_off=0.5)
    return dataset.df_filtered


@pytest.fixture(scope="module")
def stats_pipeline(regression_dataset, tmp_path_factory) -> dict:
    out_dir = tmp_path_factory.mktemp("ptp1b_regression_stats")
    stat_model = RegressionStatModel(
        dataset=regression_dataset,
        out_dir=str(out_dir),
        interaction_types_included=["Hbond", "Saltbr", "Hydrophobic", "Other"],
    )
    stat_model.calc_mutual_info_to_target()
    stat_model.calc_linear_correl_to_target()

    post_proc = StatRegressorPostProcessor(stat_model=stat_model, out_dir=str(out_dir))
    per_res = {
        "mutual_information": post_proc.get_per_res_scores(stat_method="mutual_information"),
        "linear_correlation": post_proc.get_per_res_scores(stat_method="linear_correlation"),
    }
    pymol_projections.project_pymol_top_features(
        per_feature_scores=stat_model.linear_correlations,
        model_name="linear_correlation",
        numb_features=100,
        out_dir=str(out_dir),
    )
    pymol_projections.project_pymol_per_res_scores(
        per_res_scores=per_res["linear_correlation"], model_name="linear_correlation", out_dir=str(out_dir)
    )
    return {"stat_model": stat_model, "per_res": per_res, "out_dir": out_dir}


def test_target_column_kept_through_filters(regression_dataset):
    assert regression_dataset.columns[0] == "Target"
    assert regression_dataset["Target"].notna().all()
    assert TARGET_FEATURE not in regression_dataset.columns


def test_stats_find_target_residues(stats_pipeline):
    top_correlated = list(stats_pipeline["stat_model"].linear_correlations)[:5]
    assert any("220Arg" in feat or "180Asp" in feat for feat in top_correlated)
    for method in ("mutual_information", "linear_correlation"):
        top_residues = list(stats_pipeline["per_res"][method])[:3]
        assert top_residues[0] in KEY_RESIDUES
        assert len(set(top_residues) & KEY_RESIDUES) >= 2


def test_stats_output_files(stats_pipeline):
    names = {p.name for p in stats_pipeline["out_dir"].iterdir()}
    assert {
        "Mutual_Information_Per_Feature_Scores.csv",
        "Linear_Correlations_Per_Feature_Scores.csv",
        "Mutual_Information_Scores_Per_Residue.csv",
        "Linear_Correlation_Scores_Per_Residue.csv",
        "linear_correlation_Pymol_Per_Feature_Scores.py",
        "linear_correlation_Pymol_Per_Res_Scores.py",
    } <= names
    text = (stats_pipeline["out_dir"] / "linear_correlation_Pymol_Per_Feature_Scores.py").read_text()
    assert len(re.findall(r"distance interaction\d+", text)) == 100


def test_ml_regression(regression_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "ml_results"
    ml_model = RegressionModel(
        dataset=regression_dataset,
        models_to_use=["Random_Forest"],
        out_dir=str(out_dir),
        cross_validation_splits=2,
        cross_validation_repeats=1,
        search_approach="none",
    )
    ml_model.build_models(save_models=True)
    metrics = ml_model.evaluate_models()
    assert metrics["Model"].tolist() == ["Random_Forest"]

    post_proc = SupervisedPostProcessor(out_dir=str(out_dir))
    post_proc.load_models_from_instance(supervised_model=ml_model)
    post_proc.get_per_feature_scores()
    post_proc.get_per_res_scores()

    pymol_projections.project_multiple_per_res_scores(
        all_per_res_scores=post_proc.all_per_residue_scores, out_dir=str(out_dir)
    )
    pymol_projections.project_multiple_per_feature_scores(
        all_per_feature_scores=post_proc.all_per_feature_scores, numb_features=100, out_dir=str(out_dir)
    )

    assert (out_dir / "Random_Forest_Pymol_Per_Res_Scores.py").exists()
    assert (out_dir / "Random_Forest_Pymol_Per_Feature_Scores.py").exists()
