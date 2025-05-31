"""
Creates PyMOL compatible python scripts to visualise user generated results
on a 3D model of the protein.

4 Functions available for the end user:

1. project_pymol_per_res_scores(per_res_scores, model_name, out_dir)
    Write out a PyMOL compatible python script to project the per residue scores.

2. project_multiple_per_res_scores(all_per_res_scores, out_dir)
    Write out multiple PyMOL compatible visualisation scripts for
    the per residue scores, one script for each model used.

3. project_pymol_top_features()
    Write out a PyMOL compatible python script to project the top features.

4. project_multiple_per_feature_scores(all_feature_scores, numb_features, out_dir)
    Write out multiple PyMOL compatible scripts for different models.

"""

from pathlib import Path
from typing import Union

import pandas as pd

from key_interactions_finder.project_structure_utils import (
    _extract_interaction_types,
    _extract_residue_lists,
    _scale_interaction_strengths,
    _write_file,
)
from key_interactions_finder.utils import _prep_out_dir


def project_pymol_per_res_scores(per_res_scores: dict, model_name: str = "", out_dir: str = "") -> None:
    """
    Write out a PyMOL compatible python script to project the per residue scores.

    Parameters
    ----------

    per_res_scores : dict
        The keys are each residue and values the per residue score.

    model_name : str
        Appended to start of output file to identify it.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.
    """
    out_dir = _prep_out_dir(out_dir)

    # Header
    per_res_import_out = ""
    per_res_import_out += "# You can run me in several ways, perhaps the easiest way is to:\n"
    per_res_import_out += "# 1. Load the PDB file of your system in PyMOL.\n"
    per_res_import_out += "# 2. Type: @[FILE_NAME.py] in the command line.\n"
    per_res_import_out += "# 3. Make sure the .py file is in the same directory as the pdb.\n"
    per_res_import_out += "set sphere_color, red\n"
    per_res_import_out += "# The lines below are suggestions for potentially nicer figures.\n"
    per_res_import_out += "# You can comment them in if you want.\n"
    per_res_import_out += "# bg_color white\n"
    per_res_import_out += "# set cartoon_color, grey90\n"
    per_res_import_out += "# set ray_opaque_background, 0\n"
    per_res_import_out += "# set antialias, 2\n"
    per_res_import_out += "# set ray_shadows, 0\n"

    # Main, tells PyMOL to show spheres and set their size accordingly.
    for res_numb, sphere_size in per_res_scores.items():
        per_res_import_out += f"show spheres, resi {res_numb} and name CA\n"
        per_res_import_out += f"set sphere_scale, {sphere_size:.4f}, resi {res_numb} and name CA\n"

    # user selection of all CA carbons so easy to modify the sphere colours etc...
    all_spheres_list = list(per_res_scores.keys())
    all_spheres_str = "+".join(map(str, all_spheres_list))
    per_res_import_out += f"sele All_Spheres, resi {all_spheres_str} and name CA\n"

    out_file_name = model_name + "_Pymol_Per_Res_Scores.py"
    out_file_path = Path(out_dir, out_file_name)
    _write_file(out_file_path, per_res_import_out)
    print(f"The file: {out_file_path} was written to disk.")


def project_multiple_per_res_scores(all_per_res_scores: dict, out_dir: str = "") -> None:
    """
    Write out multiple PyMOL compatible visualisation scripts for
    the per residue scores, one script for each model used.

    Parameters
    ----------

    all_per_res_scores : dict
        Nested dictionary, the outer layer keys are the model names/methods used.
        The inner layer is a dict with keys being each residue and
        values the per residue score.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.
    """
    for model_name, model_scores in all_per_res_scores.items():
        project_pymol_per_res_scores(per_res_scores=model_scores, model_name=str(model_name), out_dir=out_dir)


def project_pymol_top_features(
    per_feature_scores: dict, model_name: str, numb_features: Union[int, str] = "all", out_dir: str = ""
) -> None:
    """
    Write out a PyMOL compatible python script to project the top X features.
    Features will be shown as cylinders between each residue pair,
    with cylinder size controlled according to relative score and
    cylinder colour controlled by interaction type.

    Parameters
    ----------

    per_feature_scores : dict
        Keys are the names of the features and values are their scores.

    model_name : str
        What name to appended to the start of the output file name to help identify it.

    numb_features : int or str
        The max number of top scoring features to determine (specified by an int).
        Alternatively, if set to "all", then all feature scores will be determined.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.
    """
    out_dir = _prep_out_dir(out_dir)

    df_feat_import = df_feat_import = pd.DataFrame(per_feature_scores.items())
    df_feat_import_res = df_feat_import[0].str.split(" ", expand=True)

    res1, res2 = _extract_residue_lists(df_feat_import_res)
    interact_color = _extract_interaction_types(df_feat_import_res)
    interact_strengths = _scale_interaction_strengths(df_feat_import)

    # Header of output file.
    top_feats_out = ""
    top_feats_out += "# You can run me in several ways, perhaps the easiest way is to:\n"
    top_feats_out += "# 1. Load the PDB file of your system in PyMOL.\n"
    top_feats_out += "# 2. Type: @[FILE_NAME.py] in the command line.\n"
    top_feats_out += "# Make sure the .py files are in the same directory as the pdb.\n"
    top_feats_out += "# The lines below are suggestions for potentially nicer figures.\n"
    top_feats_out += "# You can comment them in if you want.\n"
    top_feats_out += "# bg_color white\n"
    top_feats_out += "# set cartoon_color, grey90\n"
    top_feats_out += "# set ray_opaque_background, 0\n"
    top_feats_out += "# set antialias, 2\n"
    top_feats_out += "# set ray_shadows, 0\n"

    # Main, show CA carbons as spheres and set their size.
    if numb_features == "all":
        numb_features = len(res1)
    elif not isinstance(numb_features, int):
        raise ValueError("You defined the parameter 'numb_features' as neither 'all' or as an integer.")

    # prevent issue if user requests more features than exist.
    numb_features = min(numb_features, len(res1))

    for i in range(numb_features):
        feature_rep = (
            f"distance interaction{i}, "
            + f"resid {str(res1[i])} and name CA, "
            + f"resid {str(res2[i])} and name CA \n"
            + f"set dash_radius, {interact_strengths[i]}, interaction{i} \n"
            f"set dash_color, {interact_color[i]}, interaction{i} \n"
        )
        top_feats_out += feature_rep

    # Finally, group all cylinders made together - easier for user to handle in PyMOL
    top_feats_out += "group All_Interactions, interaction* \n"

    # general cylinder settings
    top_feats_out += "set dash_gap, 0.00, All_Interactions \n"
    top_feats_out += "set dash_round_ends, off, All_Interactions \n"
    top_feats_out += "hide labels \n"

    out_file_name = model_name + "_Pymol_Per_Feature_Scores.py"
    out_file_path = Path(out_dir, out_file_name)
    _write_file(out_file_path, top_feats_out)
    print(f"The file: {out_file_path} was written to disk.")


def project_multiple_per_feature_scores(
    all_per_feature_scores: dict, numb_features: Union[int, str], out_dir: str = ""
) -> None:
    """
    Write out multiple PyMOL compatible scripts for different models.

    Parameters
    ----------

    all_per_feature_scores : dict
        Nested dictionary, the outer layer keys are the model names/methods used.
        The inner layer is a dict with keys being each residue and
        values the per residue score.

    numb_features : int or str
        The max number of top scoring features to determine (specified by an int).
        Alternatively, if set to "all", then all per feature scores will be determined.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.
    """
    for model_name, model_scores in all_per_feature_scores.items():
        project_pymol_top_features(
            per_feature_scores=model_scores, model_name=model_name, numb_features=numb_features, out_dir=out_dir
        )
