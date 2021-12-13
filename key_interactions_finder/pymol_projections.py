"""
Creates PyMOL compatable python scripts to visualise user generated results.
"""
import pandas as pd
from key_interactions_finder.utils import _prep_out_dir


def write_file(file_name, text):
    """Write out a PyMOL text file."""
    file_out = open(file_name, "w+")
    file_out.write(text)
    file_out.close()
    return None


def project_multiple_per_res_scores(all_per_res_scores, out_dir="") -> None:
    """
    Write out multiple PyMOL compatabile scripts for different models.

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
        project_pymol_per_res_scores(
            per_res_scores=model_scores,
            model_name=str(model_name),
            out_dir=out_dir
        )


def project_pymol_per_res_scores(per_res_scores, model_name, out_dir="") -> None:
    """
    Write out a PyMOL compatabile python script to project per residue scores.

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
    per_res_import_out += "bg_color white\n"
    per_res_import_out += "set sphere_color, red\n"
    per_res_import_out += "set cartoon_color, grey90\n"
    per_res_import_out += "# The below 3 lines are suggestions for potentially nicer figures.\n"
    per_res_import_out += "# set ray_opaque_background, 0 \n"
    per_res_import_out += "# set antialias, 2\n"
    per_res_import_out += "# set ray_shadows, 0\n"

    # Main, tells PyMOL to show spheres and set their size accordingly.
    for res_numb, sphere_size in per_res_scores.items():
        per_res_import_out += f"show spheres, resi {res_numb} and name CA\n"
        per_res_import_out += f"set sphere_scale, {sphere_size:.4f}, resi {res_numb} and name CA\n"
        per_res_import_out += f"sele All_Spheres, resi {res_numb} and name CA\n"

    out_file = out_dir + model_name + "_Pymol_Per_Res_Scores.py"
    write_file(out_file, per_res_import_out)
    print(f"The file: {out_file} was written to disk.")


def project_multiple_per_feature_scores(all_feature_scores,
                                        numb_features,
                                        out_dir="") -> None:
    """
    Write out multiple PyMOL compatabile scripts for different models.

    Parameters
    ----------
    all_feature_scores : dict
        Nested dictionary, the outer layer keys are the model names/methods used.
        The inner layer is a dict with keys being each residue and
        values the per residue score.

    numb_features : int or str
        The max number of top scoring features to determine (specified by an int).
        Alternatively, if set to "all", then all feature importances will be determined.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.

    """
    for model_name, model_scores in all_feature_scores.items():
        project_pymol_top_features(
            per_feature_scores=model_scores,
            model_name=model_name,
            numb_features=numb_features,
            out_dir=out_dir
        )


def project_pymol_top_features(per_feature_scores, model_name, numb_features="all", out_dir=""):
    """
    Write out a PyMOL compatabile python script to project the top X features.
    Features will be shown as cylinders between each residue pair,
    with cylinder size controlled according to relative importance and
    cylinder colour controlled by interaction type.

    Parameters
    ----------
    per_feature_scores : dict
        Keys are the names of the features and values are their importances.

    model_name : str
        What name to appended to start of output file name to identify it.

    numb_features : int or str
        The max number of top scoring features to determine (specified by an int).
        Alternatively, if set to "all", then all feature importances will be determined.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.

    """
    stick_col_scheme = {"Hbond": "red", "Saltbr": "blue",
                        "Hydrophobic": "green", "Other": "magenta"}

    out_dir = _prep_out_dir(out_dir)

    df_feat_import = df_feat_import = pd.DataFrame(per_feature_scores.items())
    df_feat_import_res = df_feat_import[0].str.split(" ", expand=True)

    residue1 = (df_feat_import_res[0].str.extract(
        "(\d+)")).astype(int).values.tolist()
    residue2 = (df_feat_import_res[1].str.extract(
        "(\d+)")).astype(int).values.tolist()
    res1 = [item for sublist in residue1 for item in sublist]
    res2 = [item for sublist in residue2 for item in sublist]

    # Define interaction type and colouring to use.
    interact_type = df_feat_import_res[2].values.tolist()
    interact_col = [stick_col_scheme[i]
                    for i in interact_type if i in stick_col_scheme]

    # Determine interaction strength value and scale so max is 0.5 (Good for PyMOL).
    interact_strengths = df_feat_import[1]
    max_strength = max(interact_strengths)
    interact_strengths_scaled = []
    for interaction in interact_strengths:
        interact_strengths_scaled.append(interaction / max_strength / 2)

    interact_strengths_scaled_rounded = [
        round(elem, 4) for elem in interact_strengths_scaled]

    # Header of output file.
    top_feats_out = ""
    top_feats_out += "bg_color white\n"
    top_feats_out += "set cartoon_color, grey90\n"
    top_feats_out += "# You can run me in several ways, perhaps the easiest way is to:\n"
    top_feats_out += "# 1. Load the PDB file of your system in PyMOL.\n"
    top_feats_out += "# 2. Download and run the draw_links.py script.\n"
    top_feats_out += "# It can be obtained from: http://pldserver1.biochem.queensu.ca/~rlc/work/pymol/draw_links.py \n"
    top_feats_out += "# 3. Type: @[FILE_NAME.py] in the command line.\n"
    top_feats_out += "# 4. Make sure the .py files are in the same directory as the pdb.\n"
    top_feats_out += "# The below 3 lines are suggestions for potentially nicer figures.\n"
    top_feats_out += "# set ray_opaque_background, 0\n"
    top_feats_out += "# set antialias, 2\n"
    top_feats_out += "# set ray_shadows, 0\n"

    # Main, show spheres and set their size.
    if numb_features == "all":
        for i, _ in enumerate(res1):
            feature_rep = f"draw_links selection1=resi {res1[i]}, " + \
                f"selection2=resi {res2[i]}, " + \
                f"color={interact_col[i]}, " + \
                f"radius={interact_strengths_scaled_rounded[i]} \n"
            top_feats_out += feature_rep

    elif isinstance(numb_features, int):
        # prevent issue if user requests more features than exist.
        max_features = min(numb_features, len(res1))
        for i in range(0, max_features):
            feature_rep = f"draw_links selection1=resi {res1[i]}, " + \
                f"selection2=resi {res2[i]}, " + \
                f"color={interact_col[i]}, " + \
                f"radius={interact_strengths_scaled_rounded[i]} \n"
            top_feats_out += feature_rep

    else:
        raise ValueError(
            "You defined the parameter 'numb_features' as neither 'all' or as an integer.")

    # Finally, group all cylinders made together - easier for user to handle in PyMOL
    top_feats_out += "group All_Features, link*\n"

    out_file = out_dir + model_name + "_Pymol_Per_Feature_Scores.py"
    write_file(out_file, top_feats_out)
    print(f"The file: {out_file} was written to disk.")
