"""
Prepares data for various form of network/correlation based analyses in different programs.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances

from key_interactions_finder.utils import _prep_out_dir, _filter_features


@dataclass
class CorrelationNetwork:
    """Descript. """

    # Generated at runtime.
    dataset: pd.core.frame.DataFrame
    out_dir: str = ""
    interaction_types_included: list = field(
        default_factory=["Hbond", "Hydrophobic", "Saltbr", "Other"])

    # Generated later.
    full_corr_matrix: pd.core.frame.DataFrame = field(init=False)
    # full_contact_map: np.ndarray = field(init=False)

    res_corr_matrix: np.ndarray = field(init=False)
    res_contact_map: np.ndarray = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Descript. """
        self.out_dir = _prep_out_dir(self.out_dir)

        try:
            self.dataset = self.dataset.drop(["Classes"], axis=1)
        except KeyError:
            pass  # If not present then dataset is from unsupervised learning.

        if sorted(self.interaction_types_included) != sorted(["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features(
                self.dataset, self.interaction_types_included)

        # Generate the full_correlation_matrix and contact map.
        self.full_corr_matrix = self.dataset.corr()
        # self.full_contact_map = self. - #  TODO. If possible/reasonable?

        # #TODO - Consider make these later instead.
        #self.res_corr_matrix = self.gen_per_res_correl_matrix()
        #self.res_contact_map = self.gen_per_res_contact_map()

    def gen_per_res_contact_map(self):
        """
        Generate a per residue contact map (matrix) that identifies whether two residues
        are in contact with each other. Two residues considered in contact with one
        another if they share an interaction (i.e. a column name).

        Returns
        ----------
        np.ndarray
            A symmetrical matrix (along diagonal) of 1s (in contact) and 0s (not in contact).

        """
        # Generate empty matrix for each residue
        last_residue = self._get_last_residue()
        res_contact_map = np.zeros((last_residue, last_residue), dtype=int)

        contact_pairs = self._get_contact_pairs()
        for res1, res2 in contact_pairs.items():
            res_contact_map[(res1-1), (res2-1)] = 1
            res_contact_map[(res2-1), (res1-1)] = 1

        # correlation of residue to itself is 1.
        np.fill_diagonal(res_contact_map, 1)
        return res_contact_map

    def gen_per_res_correl_matrix(self):
        """
        For every residue to every other residue determine the interaction (if exists)
        with the strongest correlation between them and use it to build a per residue
        correlation matrix.

        Returns
        ----------
        np.ndarray
            A symmetrical matrix (along diagonal) of correlations between each residue.
        """
        # Generate empty correlation matrix for each residue
        last_residue = self._get_last_residue()
        per_res_matrix = np.zeros((last_residue, last_residue), dtype=float)

        # Filter correlation matrix to only include columns with that residue.
        for res1 in range(1, last_residue+1):
            res1_regex_key = str(res1) + "([A-Za-z]{3})" + " "
            res1_matrix = self.full_corr_matrix.filter(
                regex=res1_regex_key, axis=1)

            if len(res1_matrix.columns) != 0:
                # Filter matrix on other axis so matrix contains only the pairs of residues.
                for res2 in range(1, last_residue+1):
                    res2_regex_key = " " + str(res2) + "([A-Za-z]{3})" + " "
                    res1_res2_matrix = res1_matrix.filter(
                        regex=res2_regex_key, axis=0)
                    if len(res1_res2_matrix) != 0:
                        correls = res1_res2_matrix.to_numpy()
                        try:
                            # prevents identical interaction being used.
                            correls = correls[correls != 1]
                            max_correl = max(
                                correls.min(), correls.max(), key=abs)
                            per_res_matrix[(res1-1), (res2-1)] = max_correl
                            per_res_matrix[(res2-1), (res1-1)] = max_correl
                        except ValueError:  # happens if array becomes empty after removing the 1s.
                            pass

        # correlation of residue to itself is 1.
        np.fill_diagonal(per_res_matrix, 1)

        return per_res_matrix

    def _get_residue_lists(self):
        """
        Given a dataframe (self.dataset) containing only PyContact features,
        extract the residue numbers for each contact. (Helper Function.)

        Returns
        ----------
        pd.core.frame.DataFrame
            1st and 2nd residue number of each contact/feature in the dataframe.
        """
        df_cols = pd.DataFrame(list(self.dataset.columns),
                               columns=["Feature_Names"])
        df_cols["Res1"] = df_cols["Feature_Names"].str.split(
            "[a-zA-Z]+").str.get(0)
        df_cols["Res2"] = df_cols["Feature_Names"].str.split(
            "[a-zA-Z]+").str.get(1)
        df_cols["Res1"] = pd.to_numeric(df_cols["Res1"])
        df_cols["Res2"] = pd.to_numeric(df_cols["Res2"])
        return df_cols[["Res1", "Res2"]]

    def _get_last_residue(self):
        """
        Given a dataframe (self.dataset) containing only PyContact features,
        find the last residue in the sequence.
        (Helper function for generating the per residue matrices,
        so one knows when to stop).

        Returns
        ----------
        int
            Largest residue number present in dataset.
        """
        df_cols = pd.DataFrame()
        df_cols[["Res1", "Res2"]] = self._get_residue_lists()
        max_res1 = df_cols["Res1"].max(axis=0)
        max_res2 = df_cols["Res2"].max(axis=0)
        return max(max_res1, max_res2)

    def _get_contact_pairs(self):
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
        contact_pairs = dict(zip(df_cols["Res1"], df_cols["Res2"]))
        return contact_pairs


def heavy_atom_contact_map_from_pdb(pdb_file: str,
                                    first_res: int,
                                    last_res: int,
                                    d_cut: Optional[float] = 4.5,
                                    out_file: Optional[str] = None,
                                    ) -> np.ndarray:
    """
    Use mdanalysis to generate a heavy atom contact map/matrix given a pdb file.

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
    res_selection = "not name H* and resid " + \
        str(first_res) + "-" + str(last_res)
    group1 = universe.select_atoms(res_selection)
    group2 = universe.select_atoms(res_selection)
    matrix_size = (last_res - first_res) + 1

    per_res_contact_map = np.zeros((matrix_size, matrix_size), dtype=int)

    for group1_idx in range(first_res, last_res+1):
        group1_selection = "resid " + str(group1_idx)
        res1 = group1.select_atoms(group1_selection)

        for group2_idx in range(first_res, last_res+1):
            group2_selection = "resid " + str(group2_idx)
            res2 = group2.select_atoms(group2_selection)

            # Determine all heavy atom distance between residue pairs.
            dist_arr = distances.distance_array(
                res1.positions, res2.positions, box=universe.dimensions)

            # Replace matrix pos with 1 if min_dist less than cutoff.
            min_dist = dist_arr.min()
            if min_dist <= d_cut:
                per_res_contact_map[(group1_idx-1), (group2_idx-1)] = 1
                per_res_contact_map[(group2_idx-1), (group1_idx-1)] = 1

    if out_file is not None:
        np.savetxt(out_file, per_res_contact_map, delimiter=" ", fmt="%.1f")
    return per_res_contact_map


# TODO - BELOW
# CHALLENGE is that numpy minimun only wants two arrays.
# So need a general approach for the matrix if want multiple pdbs.
# def heavy_atom_contact_map_from_multiple_pdbs(pdb_files: list,
#                                               first_res: int,
#                                               last_res: int,
#                                               d_cut: Optional[float] = 4.5,
#                                               out_file: Optional[str] = None,
#                                               ) -> np.ndarray:
#     """
#     Use mdanalysis to generate a heavy atom contact map/matrix given a
#     list of pdb files.

#     """
#     min_dist_contact_maps = [_min_heavy_atom_distances(
#         pdb, first_res, last_res) for pdb in pdb_files]

#     numb_matrices = len(min_dist_contact_maps)
#     print(f"Length is {numb_matrices}")

#     if numb_matrices != 2:
#         #min_dist_map = np.minimum(min_dist_contact_maps)
#         print("working on it")

#     else:
#         print("working on it too")


#     print(min_dist_contact_maps)
#     print(min_dist_contact_maps[0])


#     # get min at each position.

#     # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html

#     # https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1


# def _min_heavy_atom_distances(pdb: str,
#                               first_res: int,
#                               last_res: int,
#                               ) -> np.ndarray:
#     """
#     Given a pdb_file return a symmetrical matrix of the minimumn heavy atom distances.
#     (Helper Function)
#     """
#     universe = mda.Universe(pdb)
#     res_selection = "not name H* and resid " + \
#         str(first_res) + "-" + str(last_res)
#     group1 = universe.select_atoms(res_selection)
#     group2 = universe.select_atoms(res_selection)
#     matrix_size = (last_res - first_res) + 1

#     min_dist_contact_map = np.zeros((matrix_size, matrix_size), dtype=float)

#     for group1_idx in range(first_res, last_res+1):
#         group1_selection = "resid " + str(group1_idx)
#         res1 = group1.select_atoms(group1_selection)

#         for group2_idx in range(first_res, last_res+1):
#             group2_selection = "resid " + str(group2_idx)
#             res2 = group2.select_atoms(group2_selection)

#             # Determine all heavy atom distance between residue pairs.
#             dist_arr = distances.distance_array(
#                 res1.positions, res2.positions, box=universe.dimensions)

#             # Replace matrix pos with 1 if min_dist less than cutoff.
#             min_dist = dist_arr.min()
#             min_dist_contact_map[(group1_idx-1), (group2_idx-1)] = min_dist
#             min_dist_contact_map[(group2_idx-1), (group1_idx-1)] = min_dist

#     return min_dist_contact_map
