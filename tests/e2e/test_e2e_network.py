"""
End-to-end: contacts CSV -> unsupervised filtering -> correlation network matrices (WISP inputs).

Follows tutorials/network_analysis_tutorial/Step1_Tutorial_PTP1B_Network_Analysis.ipynb on the committed
PTP1B subset. The heavy atom contact map is checked against the output committed with that tutorial.
"""

import numpy as np
import pandas as pd
import pytest

from key_interactions_finder import network_analysis
from key_interactions_finder.data_preperation import UnsupervisedFeatureData
from tests.conftest import PTP1B_CONTACTS, PTP1B_PDB, TESTS_DIR

WISP_INPUTS = TESTS_DIR.parent / "tutorials" / "network_analysis_tutorial" / "WISP_Inputs"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    out_dir = tmp_path_factory.mktemp("wisp_inputs")

    dataset = UnsupervisedFeatureData(input_df=pd.read_csv(PTP1B_CONTACTS))
    dataset.filter_by_occupancy(min_occupancy=50)
    dataset.filter_by_interaction_type(interaction_types_included=["Hbond", "Saltbr", "Hydrophobic", "Other"])

    network = network_analysis.CorrelationNetwork(dataset=dataset.df_filtered)
    contact_matrix = network.gen_res_contact_matrix(out_file=str(out_dir / "per_res_contact_matrix.csv"))

    return {"dataset": dataset, "network": network, "contact_matrix": contact_matrix, "out_dir": out_dir}


def test_filtered_features(pipeline):
    df = pipeline["dataset"].df_filtered
    assert "Target" not in df.columns
    assert ((df != 0).mean() > 0.5).all()


def test_correlation_matrix(pipeline):
    """~20 s: gen_res_correl_matrix runs a regex filter for every residue pair."""
    out_file = pipeline["out_dir"] / "per_res_matrix.csv"
    matrix = pipeline["network"].gen_res_correl_matrix(out_file=str(out_file))
    assert matrix.shape == (298, 298)
    np.testing.assert_array_equal(matrix, matrix.T)
    assert (np.diag(matrix) == 1).all()
    assert (np.abs(matrix) <= 1).all()
    np.testing.assert_allclose(np.loadtxt(out_file), np.round(matrix, 2))


def test_contact_matrix_has_every_contact(pipeline):
    """Every feature (residue pair) in the filtered data is a contact in the matrix."""
    matrix = pipeline["contact_matrix"]
    pairs = pipeline["network"]._get_contact_pairs()
    assert len(pairs) == len(pipeline["dataset"].df_filtered.columns)
    for res1, res2 in pairs:
        assert matrix[res1 - 1, res2 - 1] == 1
        assert matrix[res2 - 1, res1 - 1] == 1
    # And nothing else apart from the diagonal.
    n_unique_pairs = len({tuple(sorted(pair)) for pair in pairs})
    assert matrix.sum() == 2 * n_unique_pairs + 298


def test_heavy_atom_contact_map_matches_tutorial_output(tmp_path):
    """Same call as the network tutorial, compared with the map committed in its WISP_Inputs folder."""
    out_file = tmp_path / "heavy_atom_contact_map.csv"
    matrix = network_analysis.heavy_atom_contact_map_from_pdb(
        pdb_file=str(PTP1B_PDB), first_res=1, last_res=298, d_cut=5, out_file=str(out_file)
    )
    expected = np.loadtxt(WISP_INPUTS / "heavy_atom_contact_map.csv")
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(np.loadtxt(out_file), expected)
