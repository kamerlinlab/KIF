"""
Random helper functions.
TODO - list functions with mini description here.
"""
import os
import pandas as pd
import numpy as np
from MDAnalysis.analysis import distances
import MDAnalysis as mda


def _prep_out_dir(out_dir: str) -> str:
    """
    Makes the folder if it doesn't exist and appends a '/' if not present at end of a string.

    Parameters
    ----------
    out_dir : str
        Name of output directory user defines.

    Returns
    ----------
    str
        Corrected name of output directory to meet standardization criteria.
    """
    if out_dir != "":
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if out_dir[-1] != "/":
        out_dir += "/"

    return out_dir


def _filter_features_by_strings(dataset: pd.DataFrame, strings_to_preserve: list) -> pd.DataFrame:
    """
    Filter features to only include those that match one of the strings
    in the list provided.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset to filter.

    strings_to_preserve : list
        List of search terms to preserve.

    Returns
    ----------
    pd.DataFrame
        Dataframe with features filtered to only include those the user wants.
    """
    # Build the search term
    str_rep = "Target|"  # supervised/unsuperivsed learning agnostic.
    for list_item in strings_to_preserve:
        str_rep += list_item + "|"

    # must remove the final "|", otherwise it will keep all columns.
    str_rep = str_rep[:-1]

    return dataset.loc[:, dataset.columns.str.contains(str_rep)]


def per_residue_distance_to_site(
    pdb_file: str,
    out_file: str,
    site_defintion: str,
    first_residue: int,
    last_residue: int
) -> np.ndarray:
    """
    Calculate the distance of the CA carbon of each residue to an mdtraj defined
    selection of a site of interest

    Uses mdtraj under the hood.

    Parameters
    ----------
    pdb_file : str
        Path to pdb file to use for distance calculation.

    out_file : str [Optional]
        Path to output file to write out per residue distances.

    site_defintion : str
        mdtraj compatable defintion of the site of interest
        (i.e. binding site, active site etc..)
        See here for examples: https://mdtraj.org/1.9.3/atom_selection.html

    first_residue : int
        First residue to measure the distance from.

    last_residue : int
        Last residue to measure the distance to.

    Returns
    ----------
    np.ndarray
        1D array of each residues distance.

    """
    # TODO, test
    universe = mda.Universe(pdb_file)

    group2 = universe.select_atoms(site_defintion)

    min_dists = []
    for residue in range(first_residue, last_residue+1):
        selection_str = "not name H* and resid " + str(residue)
        group1 = universe.select_atoms(selection_str)

        res_dist_arr = distances.distance_array(
            group1.positions, group2.positions, box=universe.dimensions)

        min_res_dist = np.round(res_dist_arr.min(), 2)
        min_dists.append(min_res_dist)

    return min_dists
