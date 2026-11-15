"""
End-to-end: contacts CSV -> filtering -> statistical analysis -> per residue scores -> PyMOL/ChimeraX files.

Follows the statistics half of tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb, on the committed
100 frame subset of the PTP1B tutorial dataset (tests/data/). The WPD-loop (residues 177-185) and Arg221
(220 in this numbering) are the known key residues for the open/closed transition.
"""

import re

import pandas as pd
import pytest

from key_interactions_finder import chimerax_projections, pymol_projections, utils
from key_interactions_finder.data_preperation import SupervisedFeatureData
from key_interactions_finder.post_proccessing import StatClassificationPostProcessor
from key_interactions_finder.stat_modelling import ClassificationStatModel
from tests.conftest import PTP1B_CLASSES, PTP1B_CONTACTS, PTP1B_PDB

INTERACTION_TYPES = ["Hbond", "Saltbr", "Hydrophobic"]
WPD_LOOP = set(range(177, 186))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    """Run the whole pipeline once (as the tutorial does) and return the objects and output folder."""
    out_dir = tmp_path_factory.mktemp("ptp1b_stats")

    # 1. Load the contacts CSV and add the per frame class labels.
    dataset = SupervisedFeatureData(
        input_df=pd.read_csv(PTP1B_CONTACTS),
        target_file=str(PTP1B_CLASSES),
        is_classification=True,
        header_present=False,
    )

    # 2. Filter.
    dataset.reset_filtering()
    dataset.filter_by_occupancy_by_class(min_occupancy=25)
    dataset.filter_by_interaction_type(interaction_types_included=INTERACTION_TYPES)
    dataset.filter_by_avg_strength(average_strength_cut_off=0.5)

    # 3. Statistical analysis.
    stat_model = ClassificationStatModel(
        dataset=dataset.df_filtered,
        class_names=["Closed", "Open"],
        out_dir=str(out_dir),
        interaction_types_included=INTERACTION_TYPES,
    )
    stat_model.calc_js_distances()
    stat_model.calc_mutual_info_to_target()

    # 4. Post-processing.
    post_proc = StatClassificationPostProcessor(stat_model=stat_model, out_dir=str(out_dir))
    per_res = {
        "jensen_shannon": post_proc.get_per_res_scores(stat_method="jensen_shannon"),
        "mutual_information": post_proc.get_per_res_scores(stat_method="mutual_information"),
    }
    with pytest.warns(UserWarning, match="very simplistic"):
        post_proc.estimate_feature_directions()

    distances_to_site = utils.per_residue_distance_to_site(
        pdb_file=str(PTP1B_PDB),
        site_defintion="not name H* and resid 180",
        first_residue=1,
        last_residue=298,
        out_file=str(out_dir / "Distance_to_D181.csv"),
    )

    # 5. Projections onto the structure.
    feature_scores = {"jensen_shannon": stat_model.js_distances, "mutual_information": stat_model.mutual_infos}
    for name in feature_scores:
        pymol_projections.project_pymol_top_features(
            per_feature_scores=feature_scores[name], model_name=name, numb_features=125, out_dir=str(out_dir)
        )
        pymol_projections.project_pymol_per_res_scores(
            per_res_scores=per_res[name], model_name=name, out_dir=str(out_dir)
        )
        chimerax_projections.project_chimerax_top_features(
            per_feature_scores=feature_scores[name],
            model_name=name,
            pdb_file=str(PTP1B_PDB),
            numb_features=10,  # each feature reloads the PDB, keep this small.
            out_dir=str(out_dir),
        )
        chimerax_projections.project_chimerax_per_res_scores(
            per_res_scores=per_res[name], model_name=name, out_dir=str(out_dir)
        )

    return {
        "dataset": dataset,
        "stat_model": stat_model,
        "post_proc": post_proc,
        "per_res": per_res,
        "feature_scores": feature_scores,
        "distances_to_site": distances_to_site,
        "out_dir": out_dir,
    }


def test_filtering(pipeline):
    df = pipeline["dataset"].df_filtered
    assert df.columns[0] == "Target"
    assert set(df["Target"]) == {"Closed", "Open", "Neither"}
    features = df.columns[1:]
    assert 300 < len(features) < 989
    assert not any(col.endswith("Other") for col in features)
    assert (df[features].mean() > 0.5).all()


def test_all_output_files_written(pipeline):
    expected = {
        "Jensen_Shannon_Per_Feature_Scores.csv",
        "Mutual_Information_Per_Feature_Scores.csv",
        "Jensen_Shannon_Distance_Scores_Per_Residue.csv",
        "Mutual_Information_Scores_Per_Residue.csv",
        "Feature_Direction_Estimates.csv",
        "Distance_to_D181.csv",
        "jensen_shannon_Pymol_Per_Feature_Scores.py",
        "jensen_shannon_Pymol_Per_Res_Scores.py",
        "jensen_shannon_ChimeraX_Per_Feature_Scores.cxc",
        "jensen_shannon_ChimeraX_Per_Res_Scores.cxc",
        "mutual_information_Pymol_Per_Feature_Scores.py",
        "mutual_information_Pymol_Per_Res_Scores.py",
        "mutual_information_ChimeraX_Per_Feature_Scores.cxc",
        "mutual_information_ChimeraX_Per_Res_Scores.cxc",
    }
    assert expected <= {p.name for p in pipeline["out_dir"].iterdir()}


@pytest.mark.parametrize("method", ["jensen_shannon", "mutual_information"])
def test_key_features_found(pipeline, method):
    """The WPD-loop / Arg220 interactions known from the tutorial come out on top."""
    top_5 = list(pipeline["feature_scores"][method])[:5]
    assert "180Asp 220Arg Saltbr" in top_5
    assert "179Pro 220Arg Hbond" in top_5


@pytest.mark.parametrize("method", ["jensen_shannon", "mutual_information"])
def test_key_residues_found(pipeline, method):
    """Arg220 is the top residue, Asp180 is close behind."""
    per_res = pipeline["per_res"][method]
    top = list(per_res)
    assert top[0] == 220
    assert per_res[220] == 1.0
    assert 180 in top[:3]


def test_per_res_csv_matches_scores(pipeline):
    saved = pd.read_csv(pipeline["out_dir"] / "Jensen_Shannon_Distance_Scores_Per_Residue.csv")
    per_res = pipeline["per_res"]["jensen_shannon"]
    assert saved["Residue Number"].tolist() == list(per_res)
    assert saved["Normalised Score"].tolist() == pytest.approx(list(per_res.values()))


def test_pymol_per_res_script_matches_scores(pipeline):
    """Each sphere in the PyMOL script is on the residue, and has the size, from the per residue scores."""
    text = (pipeline["out_dir"] / "jensen_shannon_Pymol_Per_Res_Scores.py").read_text()
    spheres = {
        int(res): float(size) for size, res in re.findall(r"set sphere_scale, ([\d.]+), resi (\d+) and name CA", text)
    }
    per_res = pipeline["per_res"]["jensen_shannon"]
    assert set(spheres) == set(per_res)
    for res, size in spheres.items():
        assert size == pytest.approx(per_res[res], abs=1e-4)


def test_pymol_top_features_script_matches_scores(pipeline):
    text = (pipeline["out_dir"] / "jensen_shannon_Pymol_Per_Feature_Scores.py").read_text()
    links = re.findall(r"distance interaction\d+, resid (\d+) and name CA, resid (\d+) and name CA", text)
    assert len(links) == 125
    top_features = list(pipeline["feature_scores"]["jensen_shannon"])[:125]
    expected = [tuple(re.findall(r"(\d+)[A-Za-z]", feat)[:2]) for feat in top_features]
    assert links == expected


def test_chimerax_scripts(pipeline):
    per_feature = (pipeline["out_dir"] / "jensen_shannon_ChimeraX_Per_Feature_Scores.cxc").read_text()
    assert per_feature.count("shape cylinder") == 10
    per_res = (pipeline["out_dir"] / "jensen_shannon_ChimeraX_Per_Res_Scores.cxc").read_text()
    assert "shape sphere center :220@CA radius 1.5000 color red" in per_res


def test_feature_directions(pipeline):
    directions = pipeline["post_proc"].feature_directions
    assert set(directions.values()) <= {"Closed", "Open"}
    # The Asp180-Arg220 salt bridge only forms with the WPD-loop closed.
    assert directions["180Asp 220Arg Saltbr"] == "Closed"


def test_kdes(pipeline):
    x_values, kdes = pipeline["post_proc"].get_kdes(number_features=10)
    assert x_values.shape == (50, 1)
    assert list(kdes["Closed"]) == list(pipeline["stat_model"].js_distances)[:10]


def test_top_residues_are_near_wpd_loop(pipeline):
    distances = pipeline["distances_to_site"]
    top_10 = list(pipeline["per_res"]["jensen_shannon"])[:10]
    assert set(top_10) & WPD_LOOP
    assert sum(distances[res] < 15 for res in top_10) >= 5
