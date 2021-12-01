"""
Prepares data for various form of network/correlation based analyses in different programs.
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from key_interactions_finder.utils import _prep_out_dir

# TODO - Add contact map from pdb functionality here:


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
            self._filter_features()

        # Generate the full_correlation_matrix and contact map.
        self.full_corr_matrix = self.dataset.corr()
        # self.full_contact_map = self. - #  TODO.

        # maybe make these later instead.
        self.res_corr_matrix = self.gen_per_res_correl_matrix()
        self.res_contact_map = self.gen_per_res_contact_map()

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

    def _filter_features(self):
        """Filter PyContact features to include only those a certain interaction type
        in the network analysis, based on user selection."""
        # Build the search term
        str_rep = ""
        for list_item in self.interaction_types_included:
            str_rep += list_item + "|"
        # must remove final "|", otherwise keeps all columns.
        str_rep = str_rep[:-1]

        # Filter.
        self.dataset = self.dataset.loc[:, self.dataset.columns.str.contains(
            str_rep)]


# https://groups.google.com/g/mdnalysis-discussion/c/KlvkdN2bjiE
# # mdanalysis is what pycontact uses...
def heavy_atom_contact_map_from_pdb(pdb_file: str, first_res: int, last_res: int, d_cut: float, out_file: str) -> np.ndarray:
    """
    Use mdanalysis to generate a heavy atom contact map/matrix given a pdb file.

    Parameters
    ----------
    pdb+file

    first_res : int
        First residue number to use for the contact map, assumed to be 1 if not provided.

    last_res : int
        Last residue number to use for the contact map.

    d_cut : float # TODO Make optional and set default.
        Distance cut-off in Å. Default is 4.5 Å.

    out_file : str
        A path to save the file to. If left empty no file saved.

    Returns
    ----------
    np.ndarray
        Numpy matrix
    """
    print(pdb_file)
    print(d_cut)
    print(first_res, last_res)

    # return
