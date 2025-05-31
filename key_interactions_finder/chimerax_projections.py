"""
Creates ChimeraX compatible scripts to visualise user generated results
on a 3D model of the protein.

4 Functions available for the end user:

1. project_chimerax_per_res_scores(per_res_scores, model_name, out_dir)
    Write out a ChimeraX compatible script to project the per residue scores.

2. project_multiple_per_res_scores(all_per_res_scores, out_dir)
    Write out multiple ChimeraX compatible visualisation scripts for
    the per residue scores, one script for each model used.

3. project_chimerax_top_features()
    Write out a ChimeraX compatible script to project the top features.

4. project_multiple_per_feature_scores(all_feature_scores, numb_features, out_dir)
    Write out multiple ChimeraX compatible scripts for different models.

"""

import warnings
from pathlib import Path
from typing import Tuple, Union

import MDAnalysis as mda
import pandas as pd

from key_interactions_finder.project_structure_utils import (
    _extract_interaction_types,
    _extract_residue_lists,
    _scale_interaction_strengths,
    _write_file,
)
from key_interactions_finder.utils import _prep_out_dir


def get_residue_coordinates(pdb_file: str, residue_number: int) -> Tuple[float, float, float]:
    """
    Get the coordinates of the CA atom of a specified residue from a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.
    residue_number : int
        The residue number.

    Returns
    ----------
    Tuple[float, float, float]
        The x, y, z coordinates of the CA atom.
    """
    # Not relevant warning message for this use case, don't want to scare users.
    warnings.filterwarnings(action="ignore", message="Element information is missing")
    u = mda.Universe(pdb_file)

    residue = u.select_atoms(f"resid {residue_number} and name CA")
    if not residue:
        raise ValueError(f"Residue number {residue_number} not found in {pdb_file}")
    return tuple(residue.positions[0])


def project_chimerax_per_res_scores(
    per_res_scores: dict,
    model_name: str = "",
    out_dir: str = "",
    sphere_color="red",
) -> None:
    """
    Write out a ChimeraX compatible script to project the per residue scores.

    Parameters
    ----------

    per_res_scores : dict
        The keys are each residue and values the per residue score.

    model_name : str
        Appended to start of output file to identify it.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.

    sphere_color : str
        Color of spheres created per residue.
    """
    out_dir = _prep_out_dir(out_dir)

    # Header
    per_res_import_out = ""
    per_res_import_out += "# 1. Load the PDB file of your system in ChimeraX \n"
    per_res_import_out += "# 2. Run this script by using the command: open /path/[FILE_NAME.cxc]\n"
    per_res_import_out += "# 3. Make sure the .cxc file is in the same directory as the pdb.\n"
    # per_res_import_out += f"open {self.pdb_file}\n"
    per_res_import_out += "# The lines below are suggestions for potentially nicer figures.\n"
    per_res_import_out += "# You can comment them in if you want.\n"
    per_res_import_out += "#set bgColor white\n"
    per_res_import_out += "#color gray target c\n"
    per_res_import_out += "#lighting soft\n"
    per_res_import_out += "#graphics silhouettes true\n"

    # Main, tells ChimeraX to show spheres and set their size accordingly.
    for res_numb, sphere_size in per_res_scores.items():
        # additional sphere_size to scale better with ChimeraX defaults
        per_res_import_out += (
            f"shape sphere center :{res_numb}@CA radius {sphere_size * 1.5:.4f} color {sphere_color}\n"
        )

    # user selection of all CA carbons so easy to modify the sphere colours etc...
    # this doesn't work the same in ChimeraX, selects residues but shapes are separate objects
    # all_spheres_list = list(per_res_scores.keys())
    # all_spheres_str = ','.join(map(str, all_spheres_list))
    # per_res_import_out += f"select :{all_spheres_str}@CA\n"

    out_file_name = model_name + "_ChimeraX_Per_Res_Scores.cxc"
    out_file_path = Path(out_dir, out_file_name)
    _write_file(out_file_path, per_res_import_out)
    print(f"The file: {out_file_path} was written to disk.")


def project_multiple_per_res_scores(all_per_res_scores: dict, out_dir: str = "") -> None:
    """
    Write out multiple ChimeraX compatible visualisation scripts for
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
        project_chimerax_per_res_scores(per_res_scores=model_scores, model_name=str(model_name), out_dir=out_dir)


def project_chimerax_top_features(
    per_feature_scores: dict,
    model_name: str,
    pdb_file: str,
    numb_features: Union[int, str] = "all",
    out_dir: str = "",
) -> None:
    """
    Write out a ChimeraX compatible script to project the top X features.
    Features will be shown as cylinders between each residue pair,
    with cylinder size controlled according to relative score and
    cylinder colour controlled by interaction type.

    Parameters
    ----------

    per_feature_scores : dict
        Keys are the names of the features and values are their scores.

    model_name : str
        What name to appended to the start of the output file name to help identify it.

    pdb_file : str
        Path to the PDB file.

    numb_features : int or str
        The max number of top scoring features to determine (specified by an int).
        Alternatively, if set to "all", then all feature scores will be determined.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.
    """
    out_dir = _prep_out_dir(out_dir)

    df_feat_import = pd.DataFrame(per_feature_scores.items())
    df_feat_import_res = df_feat_import[0].str.split(" ", expand=True)

    res1, res2 = _extract_residue_lists(df_feat_import_res)
    interact_color = _extract_interaction_types(df_feat_import_res)
    interact_strengths = _scale_interaction_strengths(df_feat_import)

    # Header of output file.
    top_feats_out = ""
    top_feats_out += "# 1. Load the PDB file of your system in ChimeraX \n"
    top_feats_out += "# 2. Run this script by using the command: open /path/[FILE_NAME.cxc]\n"
    top_feats_out += "# 3. Make sure the .cxc file is in the same directory as the pdb.\n"
    # top_feats_out += f"open {pdb_file}\n"
    top_feats_out += "# The lines below are suggestions for potentially nicer figures.\n"
    top_feats_out += "# You can comment them in if you want.\n"
    top_feats_out += "#set bgColor white\n"
    top_feats_out += "#color gray target c\n"
    top_feats_out += "#lighting soft\n"
    top_feats_out += "#graphics silhouettes true\n"

    # Main, show CA carbons as spheres and set their size.
    if numb_features == "all":
        numb_features = len(res1)
    elif not isinstance(numb_features, int):
        raise ValueError("You defined the parameter 'numb_features' as neither 'all' or as an integer.")

    # prevent issue if user requests more features than exist.
    numb_features = min(numb_features, len(res1))

    for i in range(numb_features):
        coord1 = get_residue_coordinates(pdb_file, res1[i])
        coord2 = get_residue_coordinates(pdb_file, res2[i])
        feature_rep = f"shape cylinder radius {interact_strengths[i]} fromPoint {coord1[0]},{coord1[1]},{coord1[2]} toPoint {coord2[0]},{coord2[1]},{coord2[2]} color {interact_color[i]}\n"
        top_feats_out += feature_rep

    out_file_name = model_name + "_ChimeraX_Per_Feature_Scores.cxc"
    out_file_path = Path(out_dir, out_file_name)
    _write_file(out_file_path, top_feats_out)
    print(f"The file: {out_file_path} was written to disk.")


def project_multiple_per_feature_scores(
    all_per_feature_scores: dict,
    pdb_file: str,
    numb_features: Union[int, str],
    out_dir: str = "",
) -> None:
    """
    Write out multiple ChimeraX compatible scripts for different models.

    Parameters
    ----------

    all_per_feature_scores : dict
        Nested dictionary, the outer layer keys are the model names/methods used.
        The inner layer is a dict with keys being each residue and
        values the per residue score.

    pdb_file : str
        Path to the PDB file.

    numb_features : int or str
        The max number of top scoring features to determine (specified by an int).
        Alternatively, if set to "all", then all per feature scores will be determined.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.
    """
    for model_name, model_scores in all_per_feature_scores.items():
        project_chimerax_top_features(
            per_feature_scores=model_scores,
            model_name=model_name,
            pdb_file=pdb_file,
            numb_features=numb_features,
            out_dir=out_dir,
        )
