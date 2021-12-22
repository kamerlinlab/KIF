"""
Takes a processed PyContact feature set and prepares the dataset for either
supervised/unsupervised learning.
Main responsibilities:
1. Add classification data to supervised learning datasets.
2. Offer several filtering methods for the PyContact features.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from key_interactions_finder.utils import _filter_features_by_strings


class FeatureData(ABC):
    """Abstract base class to unify the construction of the supervised and unsupervised classes."""

    @abstractmethod
    def filter_by_occupancy(self, min_occupancy):
        """Filter features such that only features with %occupancy >= the min_occupancy are kept."""

    @abstractmethod
    def filter_by_interaction_type(self, interaction_types_included):
        """Filter features/interactions to use by their type (e.g. hbond or vdws...)."""

    @abstractmethod
    def filter_by_avg_strength(self, average_strength_cut_off):
        """Filter features/interactions to use by their average strength."""

    @abstractmethod
    def filter_by_main_or_side_chain(self, main_side_chain_types_included):
        """Filter features to only certain combinations of main and side chain interactions."""


@dataclass
class SupervisedFeatureData(FeatureData):
    """
    FeatureData Class for datasets with classification data.

    Attributes
    ----------
    input_df : pd.DataFrame
        Dataframe of PyContact features to process.

    classifications_file : str
        String for path to the classification file.

    header_present : bool
        True or False, does the classifications_file have a header.

    df_feat_class : pd.DataFrame
        Dataframe generated after merging feature and classifcation data together
        but before any filtering has been performed.

    df_filtered : pd.DataFrame
        Dataframe generated after filtering. Each time a filtering method is applied, this
        dataset is updated so all filtering method performed are preserved.

    Methods
    -------
    filter_by_occupancy(min_occupancy)
        Filter features such that only features with %occupancy >= the min_occupancy are kept.

    filter_by_interaction_type(interaction_types_included)
        Filter features/interactions to use by their type (e.g. hbond or vdws...)

    filter_by_main_or_side_chain(main_side_chain_types_included)
        Filter features to only certain combinations of main and side chain interactions.

    filter_by_avg_strength(average_strength_cut_off)
        Filter features/interactions to use by their average strength.
    """
    input_df: pd.DataFrame
    classifications_file: str
    header_present: bool = True

    df_feat_class: pd.DataFrame = field(init=False)
    df_filtered: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Merge classifications to dataframe."""
        if self.header_present:
            df_class = pd.read_csv(self.classifications_file)
        else:
            df_class = pd.read_csv(self.classifications_file, header=None)

        df_class = df_class.set_axis(["Classes"], axis=1)

        if len(df_class) == len(self.input_df):
            self.df_feat_class = pd.concat([df_class, self.input_df], axis=1)
        else:
            exception_message = (f"Number of rows for classification data: {len(df_class)} \n" +
                                 f"Number of rows for PyContact data: {len(self.input_df)} \n" +
                                 "The length of your classifications file doesn't match the " +
                                 "length of your features file. If the difference is 1, " +
                                 "check if you set the 'header_present' keyword correctly."
                                 )
            raise Exception(exception_message)

        print("Your features and class datasets has been succesufully merged.")
        print("You can access this dataset through the class attribute: '.df_feat_class'.")

    def filter_by_occupancy(self, min_occupancy: float) -> pd.DataFrame:
        """
        Filter features such that only features with %occupancy >= the min_occupancy are kept.
        (%occupancy is the % of frames that have a non-zero interaction value).
        In the supervised form, %occupancy is determined for each class, meaning only
        observations from 1 class have to meet the cut-off to keep the feature.

        Parameters
        ----------
        min_occupancy : float
            Minimum %occupancy that a feature must have to be retained.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        keep_cols = ["Classes"]  # always want "Classes" present...
        try:
            for class_label in list(self.df_filtered["Classes"].unique()):
                df_single_class = self.df_filtered[(
                    self.df_filtered["Classes"] == class_label)]
                keep_cols_single_class = list(
                    (df_single_class.loc[:, (df_single_class !=
                                             0).mean() > (min_occupancy/100)]).columns
                )
                keep_cols.extend(keep_cols_single_class)

            self.df_filtered = self.df_filtered[list(
                sorted(set(keep_cols), reverse=True))]

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            for class_label in list(self.df_feat_class["Classes"].unique()):
                df_single_class = self.df_feat_class[(
                    self.df_feat_class["Classes"] == class_label)]
                keep_cols_single_class = list(
                    (df_single_class.loc[:, (df_single_class !=
                                             0).mean() > (min_occupancy/100)]).columns
                )
                keep_cols.extend(keep_cols_single_class)

            self.df_filtered = self.df_feat_class[list(
                sorted(set(keep_cols), reverse=True))]

        return self.df_filtered

    def filter_by_interaction_type(self, interaction_types_included: list) -> pd.DataFrame:
        """
        Filter features/interactions to use by their type (e.g. hbond or vdws...)

        Parameters
        ----------
        interaction_types_included : list
            Expected list items can be one or more of: "Hbond", "Saltbr", "Hydrophobic", "Other"

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_filtered,
                strings_to_preserve=interaction_types_included
            )

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_feat_class,
                strings_to_preserve=interaction_types_included
            )

        return self.df_filtered

    def filter_by_main_or_side_chain(self,
                                     main_side_chain_types_included: list
                                     ) -> pd.DataFrame:
        """
        Filter features to only certain combinations of main and side chain interactions.

        Parameters
        ----------
        main_side_chain_types_included : list
            Expected list items can be one or more of: "bb-bb", "sc-sc", "bb-sc", "sc-bb"

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_filtered,
                strings_to_preserve=main_side_chain_types_included
            )

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_feat_class,
                strings_to_preserve=main_side_chain_types_included
            )

        return self.df_filtered

    def filter_by_avg_strength(self, average_strength_cut_off: float) -> pd.DataFrame:
        """
        Filter features/interactions to use by their average strength.

        Parameters
        ----------
        average_strength_cut_off : float
            Cutoff below which features are removed from the Dataframe.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            df_just_features = self.df_filtered.drop("Classes", axis=1)
            df_features_filtered = df_just_features.loc[:, df_just_features.mean(
            ) > average_strength_cut_off]

            df_features_filtered.insert(
                0, "Classes", self.df_filtered["Classes"])
            self.df_filtered = df_features_filtered

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            df_just_features = self.df_feat_class.drop("Classes", axis=1)
            df_features_filtered = df_just_features.loc[:, df_just_features.mean(
            ) > average_strength_cut_off]

            df_features_filtered.insert(
                0, "Classes", self.df_feat_class["Classes"])
            self.df_filtered = df_features_filtered

        return self.df_filtered


@dataclass
class UnsupervisedFeautureData(FeatureData):
    """
    FeatureData Class for datasets without any classification data.

    Attributes
    ----------
    input_df : pd.DataFrame
        Dataframe of PyContact features to process.

    df_filtered : pd.DataFrame
        Dataframe generated after filtering. If multiple filtering methods are used
        this is repeatedly updated, (so all filtering method performed on it are preserved).

    Methods
    -------
    filter_by_occupancy(min_occupancy)
        Filter features such that only features with %occupancy >= the min_occupancy are kept.

    filter_by_interaction_type(interaction_types_included)
        Filter features/interactions to use by their type (e.g. hbond or vdws...)

    filter_by_main_or_side_chain(main_side_chain_types_included)
        Filter features to only certain combinations of main and side chain interactions.

    filter_by_avg_strength(average_strength_cut_off)
        Filter features/interactions to use by their average strength.
    """
    input_df: pd.DataFrame
    df_filtered: pd.DataFrame = field(init=False)

    def filter_by_occupancy(self, min_occupancy: float) -> pd.DataFrame:
        """
        Filter features such that only features with %occupancy >= the min_occupancy are included.
        (%occupancy is the % of frames that have a non-zero interaction value for a given feature).
        The supervised form has a different implementation - hence the separation.

        Parameters
        ----------
        min_occupancy : float
            Minimum percentage occupancy that a feature must have to be retained.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            self.df_filtered = self.df_filtered.loc[:, (
                self.df_filtered != 0).mean() > (min_occupancy/100)]

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = self.input_df.loc[:, (
                self.input_df != 0).mean() > (min_occupancy/100)]

        return self.df_filtered

    def filter_by_interaction_type(self, interaction_types_included: list) -> pd.DataFrame:
        """
        Filter features/interactions to use by their type (e.g. hbond or vdws...)

        Parameters
        ----------
        interaction_types_included : list
            Expected list items can be one or more of: "Hbond", "Saltbr", "Hydrophobic", "Other"

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_filtered,
                strings_to_preserve=interaction_types_included
            )

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = _filter_features_by_strings(
                dataset=self.input_df,
                strings_to_preserve=interaction_types_included
            )

        return self.df_filtered

    def filter_by_main_or_side_chain(self, main_side_chain_types_included) -> pd.DataFrame:
        """
        Filter features to only certain combinations of main and side chain interactions.

        Parameters
        ----------
        main_side_chain_types_included : list
            Expected list items can be one or more of: "bb-bb", "sc-sc", "bb-sc", "sc-bb"

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_filtered,
                strings_to_preserve=main_side_chain_types_included
            )

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = _filter_features_by_strings(
                dataset=self.input_df,
                strings_to_preserve=main_side_chain_types_included
            )

        return self.df_filtered

    def filter_by_avg_strength(self, average_strength_cut_off: float) -> pd.DataFrame:
        """
        Filter features/interactions to use by their average strength.

        Parameters
        ----------
        average_strength_cut_off : float
            Cutoff below which features are removed from the Dataframe.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        try:
            self.df_filtered = self.df_filtered.loc[:, self.df_filtered.mean(
            ) > average_strength_cut_off]

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = self.input_df.loc[:, self.input_df.mean(
            ) > average_strength_cut_off]

        return self.df_filtered
