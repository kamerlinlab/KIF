"""Tests for utils."""

import pandas as pd
import pytest

from key_interactions_finder import utils


def test_filter_features_by_strings_keeps_target():
    df = pd.DataFrame(columns=["Target", "1Ala 5Leu Hydrophobic", "2Ser 8Thr Hbond", "3Lys 9Glu Saltbr"])
    result = utils._filter_features_by_strings(df, ["Hbond"])
    assert list(result.columns) == ["Target", "2Ser 8Thr Hbond"]


def test_filter_features_by_strings_unsupervised():
    df = pd.DataFrame(columns=["1Ala 5Leu Hydrophobic", "2Ser 8Thr Hbond"])
    assert list(utils._filter_features_by_strings(df, ["Hydrophobic"]).columns) == ["1Ala 5Leu Hydrophobic"]


def test_prep_out_dir(tmp_path):
    assert utils._prep_out_dir("") == ""
    new_dir = tmp_path / "out"
    assert utils._prep_out_dir(str(new_dir)) == str(new_dir)
    assert new_dir.is_dir()
    utils._prep_out_dir(str(new_dir))  # existing folder is fine.


@pytest.mark.parametrize("side_chain_only", [False, True])
def test_per_residue_distance_to_site(ptp1b_pdb, tmp_path, side_chain_only):
    out_file = tmp_path / "dists.csv"
    dists = utils.per_residue_distance_to_site(
        pdb_file=ptp1b_pdb,
        site_defintion="not name H* and resid 180",
        first_residue=170,
        last_residue=190,
        side_chain_only=side_chain_only,
        out_file=str(out_file),
    )
    assert list(dists) == list(range(170, 191))
    assert dists[180] == 0.0  # residue 180 is the site.
    assert all(d >= 0 for d in dists.values())
    saved = pd.read_csv(out_file)
    assert saved["Residue Number"].tolist() == list(dists)


def test_per_residue_distance_side_chain_glycine(ptp1b_pdb):
    """Glycines have no side chain, so the CA is used instead (no crash)."""
    dists = utils.per_residue_distance_to_site(
        pdb_file=ptp1b_pdb, site_defintion="resid 180", first_residue=1, last_residue=298, side_chain_only=True
    )
    assert len(dists) == 298


def test_download_rejects_unknown_url(tmp_path):
    with pytest.raises(ValueError):
        utils.download_prep_tutorial_dataset(drive_url="https://example.com/not-allowed", save_dir=str(tmp_path))
