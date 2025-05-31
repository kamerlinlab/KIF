"""
Takes a processed PyContact feature set and prepares the dataset for either
supervised/unsupervised learning.

Main responsibilities:
1. Add target variable data to supervised learning datasets.
2. Offer several filtering methods for the PyContact features.

There are 2 classes for end user usage:

1. SupervisedFeatureData
    For supervised datasets where (there is a target variable)

2. UnsupervisedFeatureData
    For unsupervised datasets where (no target variable)

These classes both inherit from the class "_FeatureData", which abstracts
as much as their shared behaviour as possible.
"""

from dataclasses import dataclass, field

import pandas as pd

from key_interactions_finder.utils import _filter_features_by_strings


@dataclass
class _FeatureData:
    """
    A parent class that unifies the handling of supervised + unsupervised datasets.
    Not to be called by an end user and has no __post_init__ method.
    Both inheriting classes have their own unique __post_init__ methods.

    Attributes
    ----------

    input_df : pd.DataFrame
        Dataframe of PyContact features to process.

    df_processed : pd.DataFrame
        Dataframe generated after the class is initialised, but before any filtering
        has been performed. If the dataset has target values (supervised), these will be included
        in this dataframe.

    df_filtered : pd.DataFrame
        Dataframe generated after filtering. Each time a filtering method is applied, this
        dataset is updated so all filtering method previously performed are preserved.

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

    reset_filtering()
        Reset the filtered dataframe back to its original form.

    """

    input_df: pd.DataFrame
    df_processed: pd.DataFrame = field(init=False)
    df_filtered: pd.DataFrame = field(init=False)

    def filter_by_occupancy(self, min_occupancy: float) -> pd.DataFrame:
        """
        Filter features such that only features with %occupancy >= the min_occupancy are included.
        (%occupancy is the % of frames that have a non-zero interaction value for a given feature).

        Parameters
        ----------

        min_occupancy : float
            Minimum percentage occupancy that a feature must have to be retained.

        Returns
        -------

        pd.DataFrame
            Filtered dataframe.
        """
        # Different paths for if filtering has/hasn't been performed yet.
        # (A try except block wouldn't work here, no error raised.)
        if len(self.df_filtered) == 0:
            self.df_filtered = self.input_df.loc[:, (self.input_df != 0).mean() > (min_occupancy / 100)]

        else:
            self.df_filtered = self.df_filtered.loc[:, (self.df_filtered != 0).mean() > (min_occupancy / 100)]

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
                dataset=self.df_filtered, strings_to_preserve=interaction_types_included
            )

        except KeyError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_processed, strings_to_preserve=interaction_types_included
            )

        return self.df_filtered

    def filter_by_main_or_side_chain(self, main_side_chain_types_included: list) -> pd.DataFrame:
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
        # Test contact data has information about "bb-bb", "sc-sc", "bb-sc", "sc-bb" included.
        # Optional to have this information, but method can only be used if present...
        example_column = self.df_processed.columns[1]
        if len(example_column.split(" ")) < 4:
            error_message = """You're trying to filter interactions based on if they are
            primarily from the side or main chain but your columns don't seem to contain
            this information."""
            raise ValueError(error_message)

        try:
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_filtered, strings_to_preserve=main_side_chain_types_included
            )

        except KeyError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = _filter_features_by_strings(
                dataset=self.df_processed, strings_to_preserve=main_side_chain_types_included
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
            df_just_features = self.df_filtered.drop("Target", axis=1)
            df_features_filtered = df_just_features.loc[:, df_just_features.mean() > average_strength_cut_off]

            df_features_filtered.insert(0, "Target", self.df_filtered["Target"])
            self.df_filtered = df_features_filtered

        except KeyError:  # if no other filtering has been performed yet, follow this path.
            df_just_features = self.df_processed.drop("Target", axis=1)
            df_features_filtered = df_just_features.loc[:, df_just_features.mean() > average_strength_cut_off]

            df_features_filtered.insert(0, "Target", self.df_processed["Target"])
            self.df_filtered = df_features_filtered

        return self.df_filtered

    def reset_filtering(self):
        """Reset the filtered dataframe back to its original form."""
        self.df_filtered = self.df_processed
        return self.df_filtered


@dataclass
class SupervisedFeatureData(_FeatureData):
    """
    FeatureData Class for datasets with classification data.

    Attributes
    ----------

    input_df : pd.DataFrame
        Dataframe of PyContact features to process.

    is_classification : bool
        Select True IF the target variable is a classifications(discrete data).
        Select False IF the target variable is a regression (continous data).

    target_file : str
        String for the path to the target variable file.

    header_present : bool
        True or False, does the target_file have a header.
        Default is True.

    df_processed : pd.DataFrame
        Dataframe generated after merging feature and classifcation data together
        but before any filtering has been performed.

    df_filtered : pd.DataFrame
        Dataframe generated after filtering. Each time a filtering method is applied, this
        dataset is updated so all filtering method previously performed are preserved.

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

    reset_filtering()
        Reset the filtered dataframe back to its original form.

    filter_by_occupancy_by_class(min_occupancy)
        Special alternative to the the standard filter features by occupancy method.
        %occupancy is determined for each class (as opposed to whole dataset),
        meaning only observations from 1 class have to meet the cut-off to keep the feature.
        Only avaible to datasets with a categorical target variable (classification).
    """

    # Others are defined in parent class.
    is_classification: bool
    target_file: str
    header_present: bool = True

    def __post_init__(self):
        """Merge target data to the dataframe, make an empty df for df_filtered."""
        if self.header_present:
            df_class = pd.read_csv(self.target_file)
        else:
            df_class = pd.read_csv(self.target_file, header=None)

        df_class = df_class.set_axis(["Target"], axis=1)

        if len(df_class) == len(self.input_df):
            self.df_processed = pd.concat([df_class, self.input_df], axis=1)
        else:
            exception_message = (
                f"Number of rows for target variables data: {len(df_class)} \n"
                + f"Number of rows for PyContact data: {len(self.input_df)} \n"
                + "The length of your target variables file doesn't match the "
                + "length of your features file. If the difference is 1, "
                + "check if you set the 'header_present' keyword correctly."
            )
            raise ValueError(exception_message)

        # Empty for now until any filtering is performed
        self.df_filtered = pd.DataFrame()

        print("Your PyContact features and target variable have been succesufully merged.")
        print("You can access this dataset through the class attribute: '.df_processed'.")

    def filter_by_occupancy_by_class(self, min_occupancy: float) -> pd.DataFrame:
        """
        Special alternative to the standard filter features by occupancy method.
        As in the standard method, only features with %occupancy >= the min_occupancy are kept.
        (%occupancy is the % of frames that have a non-zero interaction value).

        However, in this approach, %occupancy is determined for each class, meaning only
        observations from 1 class have to meet the cut-off to keep the feature.

        Only available to datasets with classification (not regression) target data.

        Parameters
        ----------

        min_occupancy : float
            Minimum %occupancy that a feature must have to be retained.

        Returns
        -------

        pd.DataFrame
            Filtered dataframe.
        """
        if not self.is_classification:
            error_message = (
                "Only datasets with discrete data (i.e. for classification) can use this method. "
                + "You specified your target data was continous (i.e. for regression)."
                + "You are likely after the method: filter_by_occupancy(min_occupancy) instead."
            )
            raise TypeError(error_message)

        keep_cols = ["Target"]  # always want "Target" present...
        try:
            for class_label in list(self.df_filtered["Target"].unique()):
                df_single_class = self.df_filtered[(self.df_filtered["Target"] == class_label)]
                keep_cols_single_class = list(
                    (df_single_class.loc[:, (df_single_class != 0).mean() > (min_occupancy / 100)]).columns
                )
                keep_cols.extend(keep_cols_single_class)

            self.df_filtered = self.df_filtered[list(sorted(set(keep_cols), reverse=True))]

        except KeyError:  # if no other filtering has been performed yet, follow this path.
            for class_label in list(self.df_processed["Target"].unique()):
                df_single_class = self.df_processed[(self.df_processed["Target"] == class_label)]
                keep_cols_single_class = list(
                    (df_single_class.loc[:, (df_single_class != 0).mean() > (min_occupancy / 100)]).columns
                )
                keep_cols.extend(keep_cols_single_class)

            self.df_filtered = self.df_processed[list(sorted(set(keep_cols), reverse=True))]

        return self.df_filtered


@dataclass
class UnsupervisedFeatureData(_FeatureData):
    """
    FeatureData Class for datasets without a target varaible.

    Attributes
    ----------

    input_df : pd.DataFrame
        Dataframe of PyContact features to process.

    df_processed : pd.DataFrame
        Dataframe generated after class initialisation.

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

    reset_filtering()
        Reset the filtered dataframe back to its original form.
    """

    def __post_init__(self):
        """Initialise an empty dataframe so dataclass can be printed."""
        self.df_filtered = pd.DataFrame()

        # A little hacky, but doing this unites the supervised + unsuperivsed methods.
        # Save a lots of code duplication...
        self.df_processed = self.input_df
