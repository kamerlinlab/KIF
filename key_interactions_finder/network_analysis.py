"""
Perform correlation analysis on the PyContact interactions/features and outputs
the results in such a way that it can easily be read into various network/correlation
based analyses tools in different programs.

There is only a single class in this module (CorrelationNetwork) as this module
does not require a target variable and if you have one it does not need it.
"""

import contextlib
import re
from dataclasses import dataclass, field
from typing import Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import distances


@dataclass
class CorrelationNetwork:
    """
    Handles the correlation analysis on PyContact datasets.
    Does not require or make use of class labels (i.e., dataset can be unsupervised).

    Attributes
    ----------

    dataset : pd.DataFrame
        Input Dataframe containing all features to be studied.

    feature_corr_matrix : pd.DataFrame
        Correlation matrix for the dataset provided.

    Methods
    -------

    gen_res_contact_matrix(out_file)
        Generate a per residue contact map (matrix) that identifies whether two residues
        are in contact with each other.

    gen_res_correl_matrix(out_file)
        For every residue to every other residue determine the interaction (if exists)
        with the strongest correlation between them and use it to build a per residue
        correlation matrix.
    """

    dataset: pd.DataFrame

    # Generated during init.
    feature_corr_matrix: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Filter features and generate the full correlation matrix."""
        with contextlib.suppress(KeyError):
            self.dataset = self.dataset.drop(["Target"], axis=1)

        self.feature_corr_matrix = self.dataset.corr()
        return self.feature_corr_matrix

    def gen_res_correl_matrix(self, out_file: Optional[str] = None) -> np.ndarray:
        """
        For every residue to every other residue determine the interaction (if exists)
        with the strongest correlation between them and use it to build a per residue
        correlation matrix.

        Parameters
        ----------

        out_file : Optional[str]
            Path to save the corelation matrix to. If left empty no file is saved.

        Returns
        ----------

        np.ndarray
            A symmetrical matrix (along diagonal) of correlations between each residue.
        """
        # Generate an empty correlation matrix for all residues.
        last_residue = self._get_last_residue()
        per_res_corr_matrix = np.zeros((last_residue, last_residue), dtype=float)

        # Filter correlation matrix to only include columns with specific res number.
        for res1 in range(1, last_residue + 1):
            res1_regex_key = self._build_regex_strs(res_number=res1)
            res1_matrix = self.feature_corr_matrix.filter(regex=res1_regex_key, axis=1)

            # Filter matrix on other axis so matrix contains only the pairs of residues.
            if len(res1_matrix.columns) != 0:
                for res2 in range(res1 + 1, last_residue + 1):
                    res2_regex_key = self._build_regex_strs(res_number=res2)
                    res1_res2_matrix = res1_matrix.filter(regex=res2_regex_key, axis=0)

                    if len(res1_res2_matrix) != 0:
                        correls = res1_res2_matrix.to_numpy()

                        try:
                            # prevent identical interactions (== 1) being used.
                            correls = correls[correls != 1]
                            max_correl = max(correls.min(), correls.max(), key=abs)

                            per_res_corr_matrix[(res1 - 1), (res2 - 1)] = max_correl
                            per_res_corr_matrix[(res2 - 1), (res1 - 1)] = max_correl

                        # ValueError will happen if array becomes empty
                        # when only identical interactions were present in matrix.
                        except ValueError:
                            pass  # value stays at 0.

        # correlation of residue to itself is 1.
        np.fill_diagonal(per_res_corr_matrix, 1)

        if out_file is not None:
            np.savetxt(out_file, per_res_corr_matrix, delimiter=" ", fmt="%.2f")
            print(f"{out_file} saved to disk.")

        return per_res_corr_matrix

    def gen_res_contact_matrix(self, out_file: Optional[str] = None) -> np.ndarray:
        """
        Generate a per residue contact map (matrix) that identifies whether two residues
        are in contact with each other. Two residues considered in contact with one
        another if they share an interaction (i.e. a column name).

        Parameters
        ----------

        out_file : Optional[str]
            Path to save the corelation matrix to. If left empty no file is saved.

        Returns
        ----------

        np.ndarray
            A symmetrical matrix (along diagonal) of 1s (in contact) and 0s (not in contact).
        """
        # Generate empty matrix for each residue
        last_residue = self._get_last_residue()
        per_res_contact_matrix = np.zeros((last_residue, last_residue), dtype=int)

        contact_pairs = self._get_contact_pairs()
        for res1, res2 in contact_pairs.items():
            per_res_contact_matrix[(res1 - 1), (res2 - 1)] = 1
            per_res_contact_matrix[(res2 - 1), (res1 - 1)] = 1

        # correlation of residue to itself is 1.
        np.fill_diagonal(per_res_contact_matrix, 1)

        if out_file is not None:
            np.savetxt(out_file, per_res_contact_matrix, delimiter=" ", fmt="%.2f")
            print(f"{out_file} saved to disk.")

        return per_res_contact_matrix

    def _get_residue_lists(self) -> pd.DataFrame:
        """
        Given a dataframe (self.dataset) containing only PyContact features,
        extract the residue numbers for each contact. (Helper Function.)

        Returns
        ----------

        pd.DataFrame
            1st and 2nd residue number of each contact/feature in the dataframe.
        """
        res1_numbs, res2_numbs = [], []
        for residue_pair in list(self.dataset.columns):
            res1_info, res2_info, _ = residue_pair.split(" ")
            res1_numb = int(re.findall(r"\d+", res1_info)[0])
            res2_numb = int(re.findall(r"\d+", res2_info)[0])
            res1_numbs.append(res1_numb)
            res2_numbs.append(res2_numb)

        res_pairs_dict = {"Res1": res1_numbs, "Res2": res2_numbs}
        return pd.DataFrame(res_pairs_dict)

    def _get_last_residue(self) -> int:
        """
        Given a dataframe (self.dataset) containing only PyContact features,
        find the last residue in the sequence.
        (Helper function for generating the per residue matrices,
        so one knows when to stop).

        Returns
        ----------

        int
            Largest residue number present in the dataset.
        """
        df_cols = pd.DataFrame()
        df_cols[["Res1", "Res2"]] = self._get_residue_lists()
        max_res1 = df_cols["Res1"].max(axis=0)
        max_res2 = df_cols["Res2"].max(axis=0)
        return max(max_res1, max_res2)

    def _get_contact_pairs(self) -> dict:
        """
        Given a dataframe (self.dataset) containing only PyContact features,
        extract the pairs of residue in contact with one another.
        (Helper function to gen protein contact map).

        Returns
        ----------

        dict
            Dictionary of all interacting residue pairs.
        """
        df_cols = pd.DataFrame()
        df_cols[["Res1", "Res2"]] = self._get_residue_lists()
        contact_pairs = dict(zip(df_cols["Res1"], df_cols["Res2"], strict=True))
        return contact_pairs

    @staticmethod
    def _build_regex_strs(res_number: int) -> str:
        """
        Given a residue number, return a regex string that will match only that
        residue number in a dataframe filled with pycontact features.

        This is not as simple as it first seems as easy to catch other residues.
        E.g. If residue is 1Arg, easy to accidently catch 11Arg and so on.

        Parameters
        ----------

        res_number: int
            Residue number to create the regex string for.

        Returns
        ----------

        str
            Regex str to be used to filter a dataframe with pycontact features.
        """
        # matches if target res number is 1st residue in name
        regex_key_p1 = "(^" + str(res_number) + ")" + "([A-Za-z]{3})" + " "

        # matches if target res number is 2nd residue in name
        regex_key_p2 = " " + str(res_number) + "([A-Za-z]{3})" + " "

        return regex_key_p1 + "|" + regex_key_p2


def heavy_atom_contact_map_from_pdb(
    pdb_file: str,
    first_res: int,
    last_res: int,
    d_cut: Optional[float] = 4.5,
    out_file: Optional[str] = None,
) -> np.ndarray:
    """
    Use MDAnalysis to generate a heavy atom contact map/matrix given a single PDB file.
    If 'out_file' specified the result will be saved to disk.

    Parameters
    ----------

    pdb_file: str
        Path to PDB file to generate the contact map from.

    first_res : int
        First residue number to use for the contact map.

    last_res : int
        Last residue number to use for the contact map.

    d_cut : Optional[float]
        Distance cut-off in Å. Default is 4.5 Å.

    out_file : Optional[str]
        Path to save the contact map file to. If left empty no file is saved.

    Returns
    ----------

    np.ndarray
        Symmetrical (along diagonal) matrix of 1s (in contact) and 0s (not in contact).
    """
    universe = mda.Universe(pdb_file)
    res_selection = "not name H* and resid " + str(first_res) + "-" + str(last_res)
    group1 = universe.select_atoms(res_selection)
    group2 = universe.select_atoms(res_selection)
    matrix_size = (last_res - first_res) + 1

    per_res_contact_map = np.zeros((matrix_size, matrix_size), dtype=int)

    for group1_idx in range(first_res, last_res + 1):
        group1_selection = "resid " + str(group1_idx)
        res1 = group1.select_atoms(group1_selection)

        for group2_idx in range(group1_idx, last_res + 1):
            group2_selection = "resid " + str(group2_idx)
            res2 = group2.select_atoms(group2_selection)

            # Determine all heavy atom distance between residue pairs.
            dist_arr = distances.distance_array(res1.positions, res2.positions, box=universe.dimensions)

            # Replace matrix pos with 1 if min_dist less than cutoff.
            min_dist = dist_arr.min()
            if min_dist <= d_cut:
                per_res_contact_map[(group1_idx - 1), (group2_idx - 1)] = 1
                per_res_contact_map[(group2_idx - 1), (group1_idx - 1)] = 1

    if out_file is not None:
        np.savetxt(out_file, per_res_contact_map, delimiter=" ", fmt="%.1f")
        print(f"{out_file} saved to disk.")
    return per_res_contact_map


def heavy_atom_contact_map_from_multiple_pdbs(
    pdb_files: list,
    first_res: int,
    last_res: int,
    d_cut: Optional[float] = 4.5,
    out_file: Optional[str] = None,
) -> np.ndarray:
    """
    Use MDAnalysis to generate a heavy atom contact map/matrix given a list of PDB files.
    If 'out_file' specified the result will be saved to disk.

    Parameters
    ----------

    pdb_file: list
        Paths to PDB files to generate the contact map from.

    first_res : int
        First residue number to use for the contact map.

    last_res : int
        Last residue number to use for the contact map.

    d_cut : Optional[float]
        Distance cut-off in Å. Default is 4.5 Å.

    out_file : Optional[str]
        Path to save the contact map file to. If left empty no file is saved.

    Returns
    ----------

    np.ndarray
        Symmetrical (along diagonal) matrix of 1s (in contact) and 0s (not in contact).
    """
    res_selection = "not name H* and resid " + str(first_res) + "-" + str(last_res)

    all_universes = [mda.Universe(pdb) for pdb in pdb_files]

    all_group1s = [all_universes[idx].select_atoms(res_selection) for idx, _ in enumerate(all_universes)]
    all_group2s = [all_universes[idx].select_atoms(res_selection) for idx, _ in enumerate(all_universes)]

    matrix_size = (last_res - first_res) + 1
    per_res_contact_map = np.zeros((matrix_size, matrix_size), dtype=int)

    for group1_idx in range(first_res, last_res + 1):
        group1_selection = "resid " + str(group1_idx)
        residue_1s = [all_group1s[idx].select_atoms(group1_selection) for idx, _ in enumerate(all_group1s)]

        for group2_idx in range(group1_idx, last_res + 1):
            group2_selection = "resid " + str(group2_idx)
            residue_2s = [all_group1s[idx].select_atoms(group2_selection) for idx, _ in enumerate(all_group2s)]

            # Find the smallest distance between the residue pairs across all pdbs
            min_dist_all_pdbs = 999  # always going to be above cut-off.
            for idx, _ in enumerate(all_group1s):
                dist_arr = distances.distance_array(
                    residue_1s[idx].positions,
                    residue_2s[idx].positions,
                    box=all_universes[idx].dimensions,
                )
                min_dist = dist_arr.min()

                if min_dist < min_dist_all_pdbs:
                    min_dist_all_pdbs = min_dist

            # Replace matrix pos with 1 if min_dist_all_pdbs less than cutoff.
            if min_dist_all_pdbs <= d_cut:
                per_res_contact_map[(group1_idx - 1), (group2_idx - 1)] = 1
                per_res_contact_map[(group2_idx - 1), (group1_idx - 1)] = 1

    if out_file is not None:
        np.savetxt(out_file, per_res_contact_map, delimiter=" ", fmt="%.1f")
        print(f"{out_file} saved to disk.")

    return per_res_contact_map
