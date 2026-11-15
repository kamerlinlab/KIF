"""Tests for network_analysis: CorrelationNetwork and the heavy atom contact maps."""

import re

import numpy as np
import pandas as pd
import pytest

from key_interactions_finder.network_analysis import (
    CorrelationNetwork,
    heavy_atom_contact_map_from_multiple_pdbs,
    heavy_atom_contact_map_from_pdb,
)


@pytest.fixture
def correlated_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, 100)
    return pd.DataFrame(
        {
            "1Ala 5Leu Hydrophobic": base,
            "1Ala 9Ser Hbond": rng.uniform(0, 1, 100),
            "1Ala 12Glu Other": rng.uniform(0, 1, 100),
            "5Leu 12Glu Other": -base + rng.normal(0, 0.01, 100),  # strongly anti-correlated with feature 1.
            "11Val 21Ile Hydrophobic": rng.uniform(0, 1, 100),
        }
    )


def test_target_column_dropped(correlated_df):
    df = correlated_df.copy()
    df.insert(0, "Target", 1)
    network = CorrelationNetwork(dataset=df)
    assert "Target" not in network.dataset.columns
    assert network.feature_corr_matrix.shape == (5, 5)


def test_contact_matrix_keeps_every_partner(correlated_df):
    """Residue 1 contacts 5, 9 and 12: all three must be in the map."""
    matrix = CorrelationNetwork(dataset=correlated_df).gen_res_contact_matrix()
    assert matrix.shape == (21, 21)
    for partner in (5, 9, 12):
        assert matrix[0, partner - 1] == 1
        assert matrix[partner - 1, 0] == 1
    assert matrix[4, 11] == matrix[11, 4] == 1  # 5Leu-12Glu
    assert matrix[10, 20] == matrix[20, 10] == 1  # 11Val-21Ile
    assert (np.diag(matrix) == 1).all()
    # 5 contacts, both triangles, plus 21 diagonal entries.
    assert matrix.sum() == 5 * 2 + 21
    np.testing.assert_array_equal(matrix, matrix.T)


def test_contact_matrix_saved(correlated_df, tmp_path):
    out_file = tmp_path / "contacts.dat"
    matrix = CorrelationNetwork(dataset=correlated_df).gen_res_contact_matrix(out_file=str(out_file))
    np.testing.assert_array_equal(np.loadtxt(out_file), matrix)


def test_correlation_matrix(correlated_df, tmp_path):
    out_file = tmp_path / "correl.dat"
    matrix = CorrelationNetwork(dataset=correlated_df).gen_res_correl_matrix(out_file=str(out_file))

    assert matrix.shape == (21, 21)
    np.testing.assert_array_equal(matrix, matrix.T)
    assert (np.diag(matrix) == 1).all()
    # Residues 1 and 5 share "1Ala 5Leu" which is strongly anti-correlated with "5Leu 12Glu".
    assert matrix[0, 4] < -0.95
    # Residues with no shared interactions stay 0.
    assert matrix[1, 2] == 0
    np.testing.assert_allclose(np.loadtxt(out_file), np.round(matrix, 2))


@pytest.mark.parametrize("res_number", [1, 2, 11, 21, 100])
def test_build_regex_strs_exact_residue(res_number):
    """Residue 1 must not match 11 or 21 (and so on)."""
    regex = re.compile(CorrelationNetwork._build_regex_strs(res_number))
    columns = ["1Ala 5Leu Hydrophobic", "11Val 21Ile Hydrophobic", "2Ser 100Thr Hbond", "21Ile 31Ala Other"]
    matched = [col for col in columns if regex.search(col)]
    for col in matched:
        res1, res2 = (int(n) for n in re.findall(r"(\d+)[A-Za-z]", col)[:2])
        assert res_number in (res1, res2)
    expected = [col for col in columns if res_number in [int(n) for n in re.findall(r"(\d+)[A-Za-z]", col)[:2]]]
    assert matched == expected


def test_heavy_atom_contact_map_from_pdb(ptp1b_pdb, tmp_path):
    out_file = tmp_path / "map.dat"
    matrix = heavy_atom_contact_map_from_pdb(ptp1b_pdb, first_res=1, last_res=20, out_file=str(out_file))
    assert matrix.shape == (20, 20)
    np.testing.assert_array_equal(matrix, matrix.T)
    assert (np.diag(matrix) == 1).all()
    # Sequence neighbours are always within 4.5 A (peptide bond).
    assert all(matrix[i, i + 1] == 1 for i in range(19))
    np.testing.assert_array_equal(np.loadtxt(out_file), matrix)


def test_heavy_atom_contact_map_multiple_pdbs_same_file(ptp1b_pdb):
    single = heavy_atom_contact_map_from_pdb(ptp1b_pdb, first_res=1, last_res=15)
    multiple = heavy_atom_contact_map_from_multiple_pdbs([ptp1b_pdb, ptp1b_pdb], first_res=1, last_res=15)
    np.testing.assert_array_equal(single, multiple)


def test_heavy_atom_contact_map_larger_cutoff_has_more_contacts(ptp1b_pdb):
    small = heavy_atom_contact_map_from_pdb(ptp1b_pdb, first_res=1, last_res=15, d_cut=3.0)
    large = heavy_atom_contact_map_from_pdb(ptp1b_pdb, first_res=1, last_res=15, d_cut=8.0)
    assert large.sum() > small.sum()
    assert (large >= small).all()
