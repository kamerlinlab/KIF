"""
Random helper functions.
TODO - list functions with mini description here.
"""
from typing import Optional
import csv
import os
import pandas as pd
import numpy as np
from MDAnalysis.analysis import distances
import MDAnalysis as mda


def per_residue_distance_to_site(pdb_file: str,
                                 site_defintion: str,
                                 first_residue: int,
                                 last_residue: int,
                                 out_file: Optional[str] = None,
                                 ) -> dict:
    """
    Calculate the closest heavy atom distance of each residue to an mdtraj defined
    selection of a site of interest. You can write the results to file if desired.

    Parameters
    ----------
    pdb_file : str
        Path to pdb file to use for the distance calculation.

    site_defintion : str
        mdtraj compatable defintion of the site of interest
        (i.e. binding site, active site etc..)
        See here for examples: https://mdtraj.org/1.9.3/atom_selection.html

    first_residue : int
        First residue to measure the distance from.

    last_residue : int
        Last residue to measure the distance to.

    out_file : Optional[str]
        Path to output file to write out data.

    Returns
    ----------
    dict
        Residue numbers are the keys and minimum distances are the values.
    """
    universe = mda.Universe(pdb_file)

    group2 = universe.select_atoms(site_defintion)

    min_dists = {}
    for residue in range(first_residue, last_residue+1):
        selection_str = "not name H* and resid " + str(residue)
        group1 = universe.select_atoms(selection_str)

        res_dist_arr = distances.distance_array(
            group1.positions, group2.positions, box=universe.dimensions)

        min_res_dist = np.round(res_dist_arr.min(), 2)
        min_dists.update({residue: min_res_dist})

    if out_file is None:
        return min_dists

    with open(out_file, "w", newline="") as file_out:
        csv_out = csv.writer(file_out)
        csv_out.writerow(["Residue Number", "Minimum Distance"])
        csv_out.writerows(min_dists.items())
        print(f"{out_file} written to disk.")
    return min_dists


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
    str_rep = "Target|"  # supervised/unsuperivsed learning agnostic.
    for list_item in strings_to_preserve:
        str_rep += list_item + "|"

    # must remove the final "|", otherwise it will keep all columns.
    str_rep = str_rep[:-1]

    return dataset.loc[:, dataset.columns.str.contains(str_rep)]
