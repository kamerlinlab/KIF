"""
Note: If you are an end user, you do not need to use this module directly.

A set of functions to help with projecting the KIF results onto 3D protein structures.

These functions are used to help create the PyMOL and ChimeraX projections of
per interaction and per residue scores.
"""

from typing import Tuple

import pandas as pd


def _extract_residue_lists(input_df: pd.DataFrame) -> Tuple[list, list]:
    """
    Extract two lists of residues that are present in each feature.

    Parameters
    ----------

    input_df : pd.DataFrame
        Dataframe of feature names.

    Returns
    ----------

    res1 : list
        residue 1 for each feature.

    res2 : list
        residue 2 for each feature.
    """
    residue1 = (input_df[0].str.extract(r"(\d+)")).astype(int).values.tolist()
    residue2 = (input_df[1].str.extract(r"(\d+)")).astype(int).values.tolist()
    res1 = [item for sublist in residue1 for item in sublist]
    res2 = [item for sublist in residue2 for item in sublist]

    return res1, res2


def _write_file(file_name: str, text: str) -> None:
    """Write out a PyMOL or Chimera text file."""
    with open(file_name, "w+", encoding="utf-8") as file_out:
        file_out.write(text)


def _extract_interaction_types(input_df: pd.DataFrame) -> list:
    """
    Extract the interaction type for each feature and what colour scheme to use.

    Parameters
    ----------
    input_df : pd.DataFrame
        Dataframe of feature names.

    Returns
    ----------
    list
        List of colors to assign for each feature.
    """
    stick_col_scheme = {"Hbond": "red", "Saltbr": "blue", "Hydrophobic": "green", "Other": "magenta"}

    interact_type = input_df[2].values.tolist()
    return [stick_col_scheme[i] for i in interact_type if i in stick_col_scheme]


def _scale_interaction_strengths(input_df: pd.DataFrame) -> list:
    """
    Determine interaction strength value and scale so max is 0.5.
    (0.5 is good for both PyMOL and Chimera).

    Parameters
    ----------
    input_df : pd.DataFrame
        Dataframe of per feature scores.

    Returns
    ----------
    list
        Scaled and rounded per feature scores.
    """
    interact_strengths = input_df[1]
    max_strength = max(interact_strengths)

    interact_strengths_scaled = []
    for interaction in interact_strengths:
        interact_strengths_scaled.append(interaction / max_strength / 2)

    return [round(elem, 4) for elem in interact_strengths_scaled]
