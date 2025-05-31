"""
Random helper functions.

Functions Available:

per_residue_distance_to_site()
    Calculate the closest heavy atom distance of each residue to an mdtraj defined
    selection of a site of interest. You can write the results to file if desired.
    Optionally can choose to only calculate minimum side chain distances.

download_prep_tutorial_dataset()
    Download one of the tutorial datasets from google drive using gdown and unzip it.


### The functions below should not be called directly by an end user ###
_prep_out_dir()
    Makes the folder if it doesn't exist and appends a '/' if not present at end of a string.

_filter_features_by_strings()
    Filter features to only include those that match one of the strings in the list provided.

"""

import csv
import shutil
from pathlib import Path
from typing import Optional

import gdown
import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import distances


def per_residue_distance_to_site(
    pdb_file: str,
    site_defintion: str,
    first_residue: int,
    last_residue: int,
    side_chain_only: bool = False,
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

    side_chain_only: bool = False,
        Choose whether you want to measure the minimum distance using only the
        side chain of each residue. If true, only the side chain atoms are used.
        For glycines (no side chain), the CA of the glycine is used instead.

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

    if side_chain_only:
        for residue in range(first_residue, last_residue + 1):
            selection_str = "not backbone and not name H* and resid " + str(residue)
            group1 = universe.select_atoms(selection_str)

            res_dist_arr = distances.distance_array(group1.positions, group2.positions, box=universe.dimensions)

            try:
                min_res_dist = np.round(res_dist_arr.min(), 2)

            # catches "zero-size array to reduction operation minimum which has no identity"
            except ValueError:
                # This happens for glycines which have no side chain...
                selection_str = "name CA and resid " + str(residue)
                group1 = universe.select_atoms(selection_str)

                res_dist_arr = distances.distance_array(group1.positions, group2.positions, box=universe.dimensions)

                min_res_dist = np.round(res_dist_arr.min(), 2)

            min_dists.update({residue: min_res_dist})

    else:  # both side and main chain route.
        for residue in range(first_residue, last_residue + 1):
            selection_str = "not name H* and resid " + str(residue)
            group1 = universe.select_atoms(selection_str)

            res_dist_arr = distances.distance_array(group1.positions, group2.positions, box=universe.dimensions)

            min_res_dist = np.round(res_dist_arr.min(), 2)
            min_dists.update({residue: min_res_dist})

    if out_file is None:
        return min_dists

    with open(out_file, "w", newline="", encoding="utf-8") as file_out:
        csv_out = csv.writer(file_out)
        csv_out.writerow(["Residue Number", "Minimum Distance"])
        csv_out.writerows(min_dists.items())
        print(f"{out_file} written to disk.")
    return min_dists


def download_prep_tutorial_dataset(drive_url: str, save_dir: str) -> None:
    """
    Download one of the tutorial datasets from google drive using gdown and unzip it.

    Parameters
    ----------

    drive_url : str
        Google drive url to download the zip file from. Checked to see if a tutorial file.

    save_dir : str
        Directory to save and unpack the tutorial files.
    """
    # tutorial links
    accepted_links = [
        "https://drive.google.com/file/d/1hJbwCCuTTgI4xglwu1vXyzo-yaZJbmUY/view?usp=share_link",
        "https://drive.google.com/file/d/13SjTIbSjF4ai_-Fn1vJDvW6-nsPmCdbw/view?usp=share_link",
        "https://drive.google.com/file/d/1nbK3fw7z1hDXiGINZe-VT1SGdPbNhF07/view?usp=share_link",
        "https://drive.google.com/file/d/1sYr_9stXLrOi_SDzHYLazSZUuzjL7lK-/view?usp=share_link",
        "https://drive.google.com/file/d/10DbX12ZNPKqRIAlqs55HC8I6zLXJnP_q/view?usp=share_link",
        "https://drive.google.com/file/d/1wPH4jOFOgIlpySLMN2ebk5PWzYgsrja2/view?usp=share_link",
        "https://drive.google.com/file/d/1G4n-CXoqtt_qZtDfDXbByJtMTpbeIA4l/view?usp=share_link",
        "https://drive.google.com/file/d/1pqFUMMjt9gDYOxtkVpDmyXwKXiHp0wFi/view?usp=share_link",
    ]

    if drive_url not in accepted_links:
        raise ValueError("You seem to be trying to download a non-tutorial file, stopping for safety.")

    # Prep the save_dir.
    if save_dir != "":
        save_dir_path = Path(save_dir)
        if not save_dir_path.exists():
            Path.mkdir(save_dir_path)
    else:
        save_dir_path = Path("")

    zip_file_path = Path(save_dir_path, "tutorial_dataset.zip")

    # gdown does not accept Path objects.
    gdown.download(url=drive_url, output=str(zip_file_path), quiet=False, fuzzy=True)

    shutil.unpack_archive(filename=zip_file_path, extract_dir=save_dir_path, format="zip")

    print("Tutorial files were successfully downloaded and unzipped.")


def _prep_out_dir(out_dir: str) -> str:
    """
    Makes the folder if it doesn't exist.

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
        out_dir_path = Path(out_dir)
        if not out_dir_path.exists():
            Path.mkdir(out_dir_path)

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
