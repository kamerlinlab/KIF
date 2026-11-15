"""Tests for pymol_projections, chimerax_projections and project_structure_utils."""

import pandas as pd
import pytest

from key_interactions_finder import chimerax_projections, pymol_projections
from key_interactions_finder.project_structure_utils import (
    _extract_interaction_types,
    _extract_residue_lists,
    _scale_interaction_strengths,
)

FEATURE_SCORES = {
    "180Asp 220Arg Saltbr": 1.0,
    "182Gly 265Gln Hbond": 0.5,
    "6Phe 271Leu Hydrophobic": 0.25,
    "1Glu 5Glu Other": 0.1,
}
PER_RES_SCORES = {220: 1.0, 180: 0.5, 5: 0.25}


def _split(scores: dict) -> pd.DataFrame:
    return pd.DataFrame(scores.items())[0].str.split(" ", expand=True)


def test_extract_residue_lists():
    assert _extract_residue_lists(_split(FEATURE_SCORES)) == ([180, 182, 6, 1], [220, 265, 271, 5])


def test_extract_interaction_types():
    assert _extract_interaction_types(_split(FEATURE_SCORES)) == ["blue", "red", "green", "magenta"]


def test_scale_interaction_strengths():
    # Largest becomes 0.5.
    assert _scale_interaction_strengths(pd.DataFrame(FEATURE_SCORES.items())) == [0.5, 0.25, 0.125, 0.05]


def test_pymol_per_res_scores(tmp_path):
    pymol_projections.project_pymol_per_res_scores(PER_RES_SCORES, model_name="test", out_dir=str(tmp_path))
    text = (tmp_path / "test_Pymol_Per_Res_Scores.py").read_text()

    expected_body = (
        "show spheres, resi 220 and name CA\n"
        "set sphere_scale, 1.0000, resi 220 and name CA\n"
        "show spheres, resi 180 and name CA\n"
        "set sphere_scale, 0.5000, resi 180 and name CA\n"
        "show spheres, resi 5 and name CA\n"
        "set sphere_scale, 0.2500, resi 5 and name CA\n"
        "sele All_Spheres, resi 220+180+5 and name CA\n"
    )
    assert expected_body in text
    assert text.startswith("# You can run me in several ways")


def test_pymol_top_features(tmp_path):
    pymol_projections.project_pymol_top_features(
        FEATURE_SCORES, model_name="test", numb_features=2, out_dir=str(tmp_path)
    )
    text = (tmp_path / "test_Pymol_Per_Feature_Scores.py").read_text()

    expected_body = (
        "distance interaction0, resid 180 and name CA, resid 220 and name CA \n"
        "set dash_radius, 0.5, interaction0 \n"
        "set dash_color, blue, interaction0 \n"
        "distance interaction1, resid 182 and name CA, resid 265 and name CA \n"
        "set dash_radius, 0.25, interaction1 \n"
        "set dash_color, red, interaction1 \n"
        "group All_Interactions, interaction* \n"
    )
    assert expected_body in text
    assert "interaction2" not in text


@pytest.mark.parametrize("numb_features", ["all", 100])
def test_pymol_top_features_all_or_more_than_available(tmp_path, numb_features):
    pymol_projections.project_pymol_top_features(
        FEATURE_SCORES, model_name="test", numb_features=numb_features, out_dir=str(tmp_path)
    )
    text = (tmp_path / "test_Pymol_Per_Feature_Scores.py").read_text()
    assert text.count("distance interaction") == len(FEATURE_SCORES)


def test_pymol_top_features_bad_numb_features(tmp_path):
    with pytest.raises(ValueError):
        pymol_projections.project_pymol_top_features(FEATURE_SCORES, "test", numb_features="ten", out_dir=str(tmp_path))


def test_pymol_multiple(tmp_path):
    pymol_projections.project_multiple_per_res_scores(
        {"RF": PER_RES_SCORES, "XGB": PER_RES_SCORES}, out_dir=str(tmp_path)
    )
    pymol_projections.project_multiple_per_feature_scores(
        {"RF": FEATURE_SCORES, "XGB": FEATURE_SCORES}, numb_features="all", out_dir=str(tmp_path)
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "RF_Pymol_Per_Feature_Scores.py",
        "RF_Pymol_Per_Res_Scores.py",
        "XGB_Pymol_Per_Feature_Scores.py",
        "XGB_Pymol_Per_Res_Scores.py",
    ]


def test_out_dir_created(tmp_path):
    out_dir = tmp_path / "new_folder"
    pymol_projections.project_pymol_per_res_scores(PER_RES_SCORES, model_name="test", out_dir=str(out_dir))
    assert (out_dir / "test_Pymol_Per_Res_Scores.py").exists()


def test_chimerax_per_res_scores(tmp_path):
    chimerax_projections.project_chimerax_per_res_scores(PER_RES_SCORES, model_name="test", out_dir=str(tmp_path))
    lines = (tmp_path / "test_ChimeraX_Per_Res_Scores.cxc").read_text().splitlines()
    shapes = [line for line in lines if not line.startswith("#")]
    # Radius is score * 1.5.
    assert shapes == [
        "shape sphere center :220@CA radius 1.5000 color red",
        "shape sphere center :180@CA radius 0.7500 color red",
        "shape sphere center :5@CA radius 0.3750 color red",
    ]


def test_get_residue_coordinates(ptp1b_pdb):
    coords = chimerax_projections.get_residue_coordinates(ptp1b_pdb, 180)
    assert len(coords) == 3
    with pytest.raises(ValueError, match="not found"):
        chimerax_projections.get_residue_coordinates(ptp1b_pdb, 9999)


def test_chimerax_top_features(tmp_path, ptp1b_pdb):
    chimerax_projections.project_chimerax_top_features(
        FEATURE_SCORES, model_name="test", pdb_file=ptp1b_pdb, numb_features=2, out_dir=str(tmp_path)
    )
    lines = (tmp_path / "test_ChimeraX_Per_Feature_Scores.cxc").read_text().splitlines()
    shapes = [line for line in lines if not line.startswith("#")]
    assert len(shapes) == 2

    c180 = chimerax_projections.get_residue_coordinates(ptp1b_pdb, 180)
    c220 = chimerax_projections.get_residue_coordinates(ptp1b_pdb, 220)
    assert shapes[0] == (
        f"shape cylinder radius 0.5 fromPoint {c180[0]},{c180[1]},{c180[2]} "
        f"toPoint {c220[0]},{c220[1]},{c220[2]} color blue"
    )
    assert shapes[1].endswith("color red")


def test_chimerax_multiple(tmp_path, ptp1b_pdb):
    chimerax_projections.project_multiple_per_res_scores({"RF": PER_RES_SCORES}, out_dir=str(tmp_path))
    chimerax_projections.project_multiple_per_feature_scores(
        {"RF": FEATURE_SCORES}, pdb_file=ptp1b_pdb, numb_features=1, out_dir=str(tmp_path)
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "RF_ChimeraX_Per_Feature_Scores.cxc",
        "RF_ChimeraX_Per_Res_Scores.cxc",
    ]
