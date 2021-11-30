"""
Create PyMOL compatable python scripts to visualise user generated results.
"""
import pandas as pd
from key_interactions_finder.utils import _prep_out_dir


def write_file(file_name, text):
    """Function to write out a PyMOL text file."""
    file_out = open(file_name, "w+")
    file_out.write(text)
    file_out.close()
    return None


def project_pymol_per_res_scores(per_res_scores, out_dir=""):
    """
    Write out a PyMOL compatabile python script to project per residue scores.

    Parameters
    ----------
    per_res_scores : dict
        Nested dictionary, the outer layer keys are the model names/methods used.
        The inner layer is a dict with keys being each residue and
        values the per residue score.

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.

    Returns
    ----------
        File is saved to disk, so nothing to return.
    """
    out_dir = _prep_out_dir(out_dir)

    for model_name, model_scores in per_res_scores.items():
        # Header
        per_res_import_out = ""
        per_res_import_out += "# You can run me in several ways, perhaps the easiests is to:\n"
        per_res_import_out += "# 1. Load the PDB file of your system in PyMOL.\n"
        per_res_import_out += "# 2. Type: @[FILE_NAME.py] in the command line.\n"
        per_res_import_out += "# 3. Make sure the .py file is in the same directory as the pdb.\n"
        per_res_import_out += "bg_color white\n"
        per_res_import_out += "set sphere_color, red\n"
        per_res_import_out += "set cartoon_color, grey90\n"
        per_res_import_out += "# The below 3 lines are optional, comment them in for nicer figures.\n"
        per_res_import_out += "# set ray_opaque_background, 0 \n"
        per_res_import_out += "# set antialias, 2\n"
        per_res_import_out += "# set ray_shadows, 0\n"

        # Main, tells PyMOL to show spheres and set their size accordingly.
        for res_numb, sphere_size in model_scores.items():
            per_res_import_out += f"show spheres, resi {res_numb} and name CA\n"
            per_res_import_out += f"set sphere_scale, {sphere_size:.4f}, resi {res_numb} and name CA\n"
            per_res_import_out += f"sele All_Spheres, resi {res_numb} and name CA\n"

        out_file = out_dir + str(model_name) + "_Pymol_Per_Res_Scores.py"
        write_file(out_file, per_res_import_out)
        print(f"The file: {out_file} was written to disk.")


def project_pymol_top_features(per_feature_scores, numb_features="all", out_dir=""):
    """
    Write out a PyMOL compatabile python script to project the top X features.
    Features will be shown as cylinders between each residue pair,
    with cylinder size controlled according to relative importance and
    cylinder colour controlled by interaction type.

    Parameters
    ----------
    name : type
        description

    out_dir : str
        Folder to save outputs to, if none given, saved to current directory.

    numb_features : int or str TODO - CHECK CORRECT WAY TO WRITE OPTIONAL HERE.
        The max number of top scoring features to write out.
        An int can be used to specificy.
        Alternatively, if set to "all", then all features written out.
        Default is "all".

    Returns
    ----------
        File is saved to disk, so nothing to return.

    """
    stick_col_scheme = {"Hbond": "red", "Saltbr": "blue",
                        "Hydrophobic": "green", "Other": "magenta"}

    out_dir = _prep_out_dir(out_dir)

    for model_name, model_scores in per_feature_scores.items():
        print(model_name)

        df_feat_import = pd.DataFrame(list(model_scores))
        df_feat_import_res = df_feat_import[0].str.split(" ", expand=True)

        # Extract the residue numbers and flatten the list of lists...
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
        top_feats_out += "# You can run me in several ways, perhaps the easiests is to:\n"
        top_feats_out += "# 1. Load the PDB file of your system in PyMOL.\n"
        top_feats_out += "# 2. Download and run the draw_links.py script.\n"
        top_feats_out += "# It can be obtained from: http://pldserver1.biochem.queensu.ca/~rlc/work/pymol/draw_links.py \n"
        top_feats_out += "# 3. Type: @[FILE_NAME.py] in the command line.\n"
        top_feats_out += "# 4. Make sure the .py files are in the same directory as the pdb.\n"
        top_feats_out += "# The below 3 lines are optional, comment them in for nicer figures.\n"
        top_feats_out += "# set ray_opaque_background, 0\n"
        top_feats_out += "# set antialias, 2\n"
        top_feats_out += "# set ray_shadows, 0\n"

        # Main, show spheres and set their size.
        # print(res1)
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
            for i in range(1, max_features):
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

        out_file = out_dir + str(model_name) + "_Pymol_Per_Feature_Scores.py"
        write_file(out_file, top_feats_out)
        print(f"The file: {out_file} was written to disk.")
