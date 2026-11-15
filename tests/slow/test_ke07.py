"""
End-to-end on the full KE07 tutorial dataset (10,001 frames), checked against the published tutorial results.

Follows the statistics part of tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb (regression, the target is the
W50 side chain chi2 dihedral). Only runs with `uv run pytest --run-slow`. The dataset is downloaded to
tutorials/tutorial_datasets/ on first use.

Reference values come from the saved cell outputs of the notebook (feature and frame counts).
"""

import re

import pandas as pd
import pytest

from key_interactions_finder import chimerax_projections, pymol_projections
from key_interactions_finder.data_preperation import SupervisedFeatureData
from key_interactions_finder.post_proccessing import StatRegressorPostProcessor
from key_interactions_finder.stat_modelling import RegressionStatModel

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def results(ke07_tutorial_dir, tmp_path_factory) -> dict:
    """Run the tutorial's statistics workflow once and return the objects and counts."""
    out_dir = tmp_path_factory.mktemp("ke07_stats")

    blocks = [pd.read_csv(ke07_tutorial_dir / f"KE07_block{i}.csv") for i in range(1, 5)]
    all_contacts_df = pd.concat(blocks, join="outer", axis=1)

    just_chi1_df = pd.read_csv(ke07_tutorial_dir / "R1_5d2w_1in10_Trp50_Chi1.dat")
    just_chi1_df = just_chi1_df.set_axis(["W50Chi1"], axis=1)
    chi1_df = pd.concat([just_chi1_df, all_contacts_df], axis=1)

    dataset = SupervisedFeatureData(
        input_df=chi1_df,
        target_file=str(ke07_tutorial_dir / "R1_5d2w_1in10_Trp50_Chi2.dat"),
        is_classification=False,
        header_present=True,
    )
    counts = {"start": len(dataset.df_processed.columns) - 1}
    dataset.reset_filtering()
    dataset.filter_by_occupancy(min_occupancy=25)
    counts["occupancy"] = len(dataset.df_filtered.columns) - 1
    dataset.filter_by_avg_strength(average_strength_cut_off=0.5)
    counts["avg_strength"] = len(dataset.df_filtered.columns) - 1

    # Remove all frames with W50 Chi1 < 160 or > 240, then drop the W50Chi1 column.
    chi1_chi2_df = dataset.df_filtered
    filtered_df = chi1_chi2_df[(chi1_chi2_df.W50Chi1 > 160) & (chi1_chi2_df.W50Chi1 < 240)]
    df_ready = filtered_df.drop("W50Chi1", axis=1).reset_index(drop=True)
    counts["rows_before"] = len(chi1_chi2_df)
    counts["rows_after"] = len(df_ready)

    stat_model = RegressionStatModel(
        dataset=df_ready,
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

    feature_scores = {
        "linear_correlation": stat_model.linear_correlations,
        "mutual_information": stat_model.mutual_infos,
    }
    pdb_file = str(ke07_tutorial_dir / "R1_5d2w.pdb")
    for name in feature_scores:
        pymol_projections.project_pymol_top_features(
            per_feature_scores=feature_scores[name], model_name=name, numb_features=100, out_dir=str(out_dir)
        )
        pymol_projections.project_pymol_per_res_scores(
            per_res_scores=per_res[name], model_name=name, out_dir=str(out_dir)
        )
        chimerax_projections.project_chimerax_top_features(
            per_feature_scores=feature_scores[name],
            model_name=name,
            pdb_file=pdb_file,
            numb_features=10,  # each feature reloads the PDB, keep this small.
            out_dir=str(out_dir),
        )
        chimerax_projections.project_chimerax_per_res_scores(
            per_res_scores=per_res[name], model_name=name, out_dir=str(out_dir)
        )

    return {
        "counts": counts,
        "df_ready": df_ready,
        "stat_model": stat_model,
        "per_res": per_res,
        "out_dir": out_dir,
    }


def test_filter_counts_match_tutorial(results):
    counts = results["counts"]
    assert (counts["start"], counts["occupancy"], counts["avg_strength"]) == (840, 830, 684)


def test_frame_filter_matches_tutorial(results):
    assert (results["counts"]["rows_before"], results["counts"]["rows_after"]) == (10001, 9281)
    assert results["df_ready"].columns[0] == "Target"
    assert "W50Chi1" not in results["df_ready"].columns


def test_scores_cover_all_features(results):
    stat_model = results["stat_model"]
    n_features = len(results["df_ready"].columns) - 1
    assert len(stat_model.linear_correlations) == n_features
    assert len(stat_model.mutual_infos) == n_features
    for per_res in results["per_res"].values():
        assert max(per_res.values()) == 1.0


def test_output_files(results):
    names = {p.name for p in results["out_dir"].iterdir()}
    for name in ("linear_correlation", "mutual_information"):
        assert {
            f"{name}_Pymol_Per_Feature_Scores.py",
            f"{name}_Pymol_Per_Res_Scores.py",
            f"{name}_ChimeraX_Per_Feature_Scores.cxc",
            f"{name}_ChimeraX_Per_Res_Scores.cxc",
        } <= names
    assert {
        "Linear_Correlations_Per_Feature_Scores.csv",
        "Mutual_Information_Per_Feature_Scores.csv",
        "Linear_Correlation_Scores_Per_Residue.csv",
        "Mutual_Information_Scores_Per_Residue.csv",
    } <= names


def test_pymol_per_res_script_matches_scores(results):
    text = (results["out_dir"] / "linear_correlation_Pymol_Per_Res_Scores.py").read_text()
    spheres = re.findall(r"set sphere_scale, ([\d.]+), resi (\d+) and name CA", text)
    per_res = results["per_res"]["linear_correlation"]
    assert [int(res) for _, res in spheres] == list(per_res)
    assert [float(size) for size, _ in spheres] == pytest.approx(list(per_res.values()), abs=1e-4)
