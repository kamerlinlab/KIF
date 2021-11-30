"""
Prepares data for various form of network/correlation based analyses in different programs.
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from key_interactions_finder.utils import _prep_out_dir

# TODO - Add contact map from pdb functionality here:
# https://groups.google.com/g/mdnalysis-discussion/c/KlvkdN2bjiE
# # mdanalysis is what pycontact uses...


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
    # full_contact_map: pd.core.frame.DataFrame = field(init=False) # TODO - wrong type, numpy
    per_res_corr_matrix: pd.core.frame.DataFrame = field(init=False)
    # per_res_contact_map: pd.core.frame.DataFrame = field(init=False) # TODO - wrong type, numpy

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

        # Is this bad practice to do this? - TODO.
        self.per_res_corr_matrix = pd.DataFrame()
        # self.per_res_contact_map = self.gen_per_res_contact_map() # Make function - TODO.

    def gen_per_res_contact_map(self):
        """
        Generate a per residue contact map (matrix) that identifies whether two residues
        are in contact with one another (1 if yes, 0 if no).
        Two residues considered in contact if they share an interaction (i.e. a column name).

        Return a numpy matrix with 1s (in contact) or 0s (not in contact).
        """
        # Generate empty matrix for each residue
        last_residue = self._get_last_residue()
        contact_matrix = np.zeros((last_residue, last_residue), dtype=int)

        contact_pairs = self._get_contact_pairs()
        for res1, res2 in contact_pairs.items():
            contact_matrix[(res1-1), (res2-1)] = 1
            contact_matrix[(res2-1), (res1-1)] = 1

        # correlation of residue to itself is 1.
        np.fill_diagonal(contact_matrix, 1)
        return contact_matrix

    def _get_residue_lists(self):
        """
        Given a df of PyContact features, extract the residue names
        from each feature as two seperate pandas series.

        Parameters
        ----------
        name : pd.core.frame.DataFrame
            Dataframe containing all features to analyse and no class column.

        Returns
        ----------
        df_cols["Res1", "Res2"] : pd.core.frame.DataFrame
            1st and 2nd residue number of each contact/feature in the dataframe.
        """
        df_cols = pd.DataFrame(list(self.dataset.columns),
                               columns=["Feature_Names"])
        df_cols["Res1"] = df_cols["Feature_Names"].str.split(
            "[a-zA-Z]+").str.get(1)
        df_cols["Res2"] = df_cols["Feature_Names"].str.split(
            "[a-zA-Z]+").str.get(2)
        df_cols["Res1"] = pd.to_numeric(df_cols["Res1"])
        df_cols["Res2"] = pd.to_numeric(df_cols["Res2"])
        return df_cols[["Res1", "Res2"]]

    def _get_last_residue(self):
        """Given a df with pycontact features, find the last residue in the sequence.
        This is used as helper function to gen the per residue correlation matrices.
        Output is an int of the residue number."""
        df_cols = pd.DataFrame()
        df_cols[["Res1", "Res2"]] = self._get_residue_lists()
        last_residue = max([df_cols["Res1"], df_cols["Res2"]].max(axis=0))
        return last_residue

    def _get_contact_pairs(self):
        """Given a df with pycontact features, extract the pairs of residue in contact.
        This is used as helper function to gen the protein contact map.
        Output is an dict of each residue."""
        df_cols = pd.DataFrame()
        df_cols[["Res1", "Res2"]] = self._get_residue_lists()
        contact_pairs = dict(zip(df_cols["Res1"], df_cols["Res2"]))
        return contact_pairs

    def _filter_features(self):
        """Filter PyContact features to include only those a certain interaction type
        in the network analysis."""
        # Build the search term
        str_rep = ""
        for list_item in self.interaction_types_included:
            str_rep += list_item + "|"
        str_rep = str_rep[:-1]

        # Filter.
        self.dataset = self.dataset.loc[:, self.dataset.columns.str.contains(
            str_rep)]
