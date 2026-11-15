"""
End-to-end ml part of the PTP1B tutorial, classification of open/closed states from contacts.

Uses the commited 100 frame subset of the PTP1B tutorial dataset.
Cross validation is cut down (2-fold, 1 repeat) for speed.
"""

import re

import pandas as pd
import pytest

from key_interactions_finder import chimerax_projections, pymol_projections
from key_interactions_finder.data_preperation import SupervisedFeatureData
from key_interactions_finder.model_building import ClassificationModel
from key_interactions_finder.post_proccessing import SupervisedPostProcessor
from tests.conftest import PTP1B_CLASSES, PTP1B_CONTACTS, PTP1B_PDB

MODELS = ["CatBoost", "XGBoost", "Random_Forest"]
WPD_LOOP_REGEX = "176|177|178|179|180|181|182|183|184|185|186|187|188|189"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    work_dir = tmp_path_factory.mktemp("ptp1b_ml")
    out_dir = work_dir / "ml_results"

    with pytest.MonkeyPatch.context() as mp:
        # build_models() writes ./temporary_files, CatBoost writes ./catboost_info.
        mp.chdir(work_dir)

        dataset = SupervisedFeatureData(
            input_df=pd.read_csv(PTP1B_CONTACTS),
            target_file=str(PTP1B_CLASSES),
            is_classification=True,
            header_present=False,
        )
        dataset.reset_filtering()
        dataset.filter_by_occupancy_by_class(min_occupancy=25)
        dataset.filter_by_interaction_type(interaction_types_included=["Hbond", "Saltbr", "Hydrophobic"])
        dataset.filter_by_avg_strength(average_strength_cut_off=0.5)

        # just like in the tutorial, we remove the WPD-loop, otherwise the classification is too easy
        df = dataset.df_filtered
        df_no_wpd_loop = df.loc[:, ~df.columns.str.contains(WPD_LOOP_REGEX)]

        ml_model = ClassificationModel(
            dataset=df_no_wpd_loop,
            evaluation_split_ratio=0.15,
            classes_to_use=["Closed", "Open"],
            models_to_use=MODELS,
            scaling_method="min_max",
            out_dir=str(out_dir),
            cross_validation_splits=2,
            cross_validation_repeats=1,
            search_approach="none",
        )
        build_results = ml_model.build_models(save_models=True)
        reports = ml_model.evaluate_models()
        confusion_matrices = ml_model.generate_confusion_matrix()

        post_proc = SupervisedPostProcessor(out_dir=str(out_dir))
        post_proc.load_models_from_instance(supervised_model=ml_model)
        post_proc.get_per_feature_scores()
        post_proc.get_per_res_scores()

        pymol_projections.project_multiple_per_res_scores(
            all_per_res_scores=post_proc.all_per_residue_scores, out_dir=str(out_dir)
        )
        pymol_projections.project_multiple_per_feature_scores(
            all_per_feature_scores=post_proc.all_per_feature_scores, numb_features="all", out_dir=str(out_dir)
        )
        chimerax_projections.project_multiple_per_res_scores(
            all_per_res_scores=post_proc.all_per_residue_scores, out_dir=str(out_dir)
        )
        chimerax_projections.project_multiple_per_feature_scores(
            all_per_feature_scores=post_proc.all_per_feature_scores,
            pdb_file=str(PTP1B_PDB),
            numb_features=5,
            out_dir=str(out_dir),
        )

    return {
        "ml_model": ml_model,
        "df_no_wpd_loop": df_no_wpd_loop,
        "build_results": build_results,
        "reports": reports,
        "confusion_matrices": confusion_matrices,
        "post_proc": post_proc,
        "work_dir": work_dir,
        "out_dir": out_dir,
    }


def test_wpd_loop_removed(pipeline):
    for col in pipeline["df_no_wpd_loop"].columns[1:]:
        res1, res2 = (int(n) for n in re.findall(r"(\d+)[A-Za-z]", col)[:2])
        assert not {res1, res2} & set(range(176, 190))


def test_only_requested_classes_used(pipeline):
    assert set(pipeline["ml_model"].dataset["Target"]) == {"Closed", "Open"}


def test_models_built_and_evaluated(pipeline):
    results = pipeline["build_results"]
    assert results["model"].tolist() == MODELS

    for model in MODELS:
        report = pipeline["reports"][model]
        assert {"Closed", "Open", "accuracy"} <= set(report.index)
        assert pipeline["confusion_matrices"][model].shape == (2, 2)


def test_models_saved_and_reloadable(pipeline):
    temp_folder = pipeline["work_dir"] / "temporary_files"
    for model in MODELS:
        assert (temp_folder / f"{model}_Model.pickle").exists()

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(pipeline["work_dir"])
        from_disk = SupervisedPostProcessor()
        from_disk.load_models_from_disk(models_to_use=MODELS)
        from_disk.get_per_feature_scores(save_result=False)
    assert from_disk.all_per_feature_scores == pipeline["post_proc"].all_per_feature_scores


def test_per_feature_and_residue_scores(pipeline):
    post_proc = pipeline["post_proc"]
    n_features = len(pipeline["df_no_wpd_loop"].columns) - 1
    for model in MODELS:
        feature_scores = post_proc.all_per_feature_scores[model]
        assert len(feature_scores) == n_features
        values = list(feature_scores.values())
        assert values == sorted(values, reverse=True)

        per_res = post_proc.all_per_residue_scores[model]
        assert max(per_res.values()) == 1.0
        assert not set(res for res, score in per_res.items() if score > 0) & set(range(176, 190))


def test_output_files(pipeline):
    names = {p.name for p in pipeline["out_dir"].iterdir()}
    for model in MODELS:
        assert {
            f"{model}_Feature_Scores.csv",
            f"{model}_Per_Residue_Scores.csv",
            f"{model}_Pymol_Per_Res_Scores.py",
            f"{model}_Pymol_Per_Feature_Scores.py",
            f"{model}_ChimeraX_Per_Res_Scores.cxc",
            f"{model}_ChimeraX_Per_Feature_Scores.cxc",
        } <= names


def test_pymol_per_res_script_matches_scores(pipeline):
    for model in MODELS:
        text = (pipeline["out_dir"] / f"{model}_Pymol_Per_Res_Scores.py").read_text()
        spheres = re.findall(r"set sphere_scale, ([\d.]+), resi (\d+) and name CA", text)
        per_res = pipeline["post_proc"].all_per_residue_scores[model]
        assert [int(res) for _, res in spheres] == list(per_res)
        assert [float(size) for size, _ in spheres] == pytest.approx(list(per_res.values()), abs=1e-4)
