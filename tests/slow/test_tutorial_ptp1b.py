"""
End-to-end on the full PTP1B tutorial dataset (10,000 frames), checked against published tutorial results.

Only runs with `uv run pytest --run-slow`. The dataset is downloaded to tutorials/tutorial_datasets/
on first use.

Reference values come from:
- The saved cell outputs of Tutorial_PTP1B_Classification_ML_Stats.ipynb (feature counts after filtering).
- The tutorial's Jensen-Shannon per feature scores.
- The matrices committed in tutorials/network_analysis_tutorial/WISP_Inputs/.
"""

import numpy as np
import pandas as pd
import pytest

from key_interactions_finder import network_analysis, pymol_projections
from key_interactions_finder.data_preperation import SupervisedFeatureData, UnsupervisedFeatureData
from key_interactions_finder.post_proccessing import StatClassificationPostProcessor
from key_interactions_finder.stat_modelling import ClassificationStatModel
from tests.conftest import TESTS_DIR

pytestmark = pytest.mark.slow

WISP_INPUTS = TESTS_DIR.parent / "tutorials" / "network_analysis_tutorial" / "WISP_Inputs"
INTERACTION_TYPES = ["Hbond", "Saltbr", "Hydrophobic"]


@pytest.fixture(scope="module")
def all_contacts_df(ptp1b_tutorial_dir) -> pd.DataFrame:
    blocks = [pd.read_csv(ptp1b_tutorial_dir / f"PTP1B_block{i}.csv") for i in range(1, 5)]
    return pd.concat(blocks, join="outer", axis=1)


@pytest.fixture(scope="module")
def stats_pipeline(all_contacts_df, ptp1b_tutorial_dir, tmp_path_factory) -> dict:
    out_dir = tmp_path_factory.mktemp("ptp1b_tutorial_stats")
    dataset = SupervisedFeatureData(
        input_df=all_contacts_df,
        target_file=str(ptp1b_tutorial_dir / "WT_PTP1B_Class_Assingments.txt"),
        is_classification=True,
        header_present=False,
    )
    counts = {"start": len(dataset.df_processed.columns) - 1}
    dataset.reset_filtering()
    dataset.filter_by_occupancy_by_class(min_occupancy=25)
    counts["occupancy"] = len(dataset.df_filtered.columns) - 1
    dataset.filter_by_interaction_type(interaction_types_included=INTERACTION_TYPES)
    counts["interaction_type"] = len(dataset.df_filtered.columns) - 1
    dataset.filter_by_avg_strength(average_strength_cut_off=0.5)
    counts["avg_strength"] = len(dataset.df_filtered.columns) - 1

    stat_model = ClassificationStatModel(
        dataset=dataset.df_filtered,
        class_names=["Closed", "Open"],
        out_dir=str(out_dir),
        interaction_types_included=INTERACTION_TYPES,
    )
    stat_model.calc_js_distances()

    post_proc = StatClassificationPostProcessor(stat_model=stat_model, out_dir=str(out_dir))
    js_per_res = post_proc.get_per_res_scores(stat_method="jensen_shannon")
    pymol_projections.project_pymol_per_res_scores(
        per_res_scores=js_per_res, model_name="jensen_shannon", out_dir=str(out_dir)
    )

    return {"counts": counts, "stat_model": stat_model, "js_per_res": js_per_res, "out_dir": out_dir}


def test_filter_counts_match_tutorial(stats_pipeline):
    assert stats_pipeline["counts"] == {"start": 989, "occupancy": 987, "interaction_type": 441, "avg_strength": 391}


def test_js_per_feature_scores_match_tutorial(stats_pipeline):
    top_4 = list(stats_pipeline["stat_model"].js_distances.items())[:4]
    assert top_4 == [
        ("180Asp 220Arg Saltbr", pytest.approx(0.99828, abs=1e-5)),
        ("182Gly 265Gln Hbond", pytest.approx(0.99765, abs=1e-5)),
        ("179Pro 220Arg Hbond", pytest.approx(0.99326, abs=1e-5)),
        ("181Phe 262Thr Hbond", pytest.approx(0.94822, abs=1e-5)),
    ]


def test_js_per_residue_scores(stats_pipeline):
    top_3 = list(stats_pipeline["js_per_res"].items())[:3]
    assert top_3 == [(220, 1.0), (180, pytest.approx(0.932, abs=1e-5)), (179, pytest.approx(0.62949, abs=1e-5))]
    text = (stats_pipeline["out_dir"] / "jensen_shannon_Pymol_Per_Res_Scores.py").read_text()
    assert "set sphere_scale, 1.0000, resi 220 and name CA" in text


@pytest.fixture(scope="module")
def network(all_contacts_df) -> network_analysis.CorrelationNetwork:
    dataset = UnsupervisedFeatureData(input_df=all_contacts_df)
    dataset.filter_by_occupancy(min_occupancy=50)
    dataset.filter_by_interaction_type(interaction_types_included=["Hbond", "Saltbr", "Hydrophobic", "Other"])
    return network_analysis.CorrelationNetwork(dataset=dataset.df_filtered)


def test_correlation_matrix_matches_wisp_input(network):
    matrix = network.gen_res_correl_matrix()
    expected = np.loadtxt(WISP_INPUTS / "per_res_matrix.csv")
    np.testing.assert_array_equal(np.round(matrix, 2), expected)


def test_multi_pdb_contact_map_matches_wisp_input(ptp1b_tutorial_dir):
    matrix = network_analysis.heavy_atom_contact_map_from_multiple_pdbs(
        pdb_files=[
            str(ptp1b_tutorial_dir / "WT_PTP1B_Phospho_Enzyme_Closed.pdb"),
            str(ptp1b_tutorial_dir / "WT_PTP1B_Phospho_Enzyme_Open.pdb"),
        ],
        first_res=1,
        last_res=298,
        d_cut=6,
    )
    np.testing.assert_array_equal(matrix, np.loadtxt(WISP_INPUTS / "heavy_atom_contact_map_MultiPDB.csv"))
