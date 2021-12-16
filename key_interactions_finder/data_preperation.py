"""
Add docstring here.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from key_interactions_finder.utils import _filter_features_by_strings


class FeatureData(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised classes."""

    @abstractmethod
    def filter_features_by_occupancy(self, min_occupancy):
        """Filter features/interactions to use by a minimum allowed observation cut-off."""

    @abstractmethod
    def filter_features_by_type(self, interaction_types_included):
        """Filter features/interactions to use by their type (e.g. hbond or vdws...)"""

    @abstractmethod
    def filter_features_by_avg_strength(self, average_strength_cut_off):
        """Filter features/interactions to use by their average strength."""

    @abstractmethod
    def filter_features_by_main_or_side_chain(self, main_side_chain_types_included):
        """Filter features/interactions to only certain combinations of main and side chain."""


@dataclass
class UnsupervisedFeautureData(FeatureData):
    """FeatureData Class without any classification data."""

    input_df: pd.DataFrame
    df_filtered: pd.DataFrame = field(init=False)

    def filter_features_by_occupancy(self, min_occupancy):
        """
        Filter features such that only features with %occupancy >= the min_occupancy are included.
        (%occupancy is the % of frames that have a non-zero interaction value for a given feature).
        The supervised form has a different implementation - hence the seperation.

        Parameters
        ----------
        min_occupancy : int
            Minimun percantage occupancy that a feature must have to be retained.

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

    def filter_features_by_type(self, interaction_types_included):
        """Filter features/interactions to use by their type (e.g. hbond or vdws...)"""
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

    def filter_features_by_main_or_side_chain(self, main_side_chain_types_included):
        """Filter features/interactions to only certain combinations of main and side chain."""
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

    def filter_features_by_avg_strength(self, average_strength_cut_off):
        """Filter features/interactions to use by their average strength."""
        try:
            self.df_filtered = self.df_filtered.loc[:, self.df_filtered.mean(
            ) > average_strength_cut_off]

        except AttributeError:  # if no other filtering has been performed yet, follow this path.
            self.df_filtered = self.input_df.loc[:, self.input_df.mean(
            ) > average_strength_cut_off]

        return self.df_filtered


@dataclass
class SupervisedFeatureData(FeatureData):
    """FeatureData Class with classification data included."""

    input_df: pd.DataFrame
    classifications_file: str
    header_present: bool = True

    df_feat_class: pd.DataFrame = field(init=False)
    df_filtered: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Merge per frame classification results to dataframe."""
        if self.header_present:
            df_class = pd.read_csv(self.classifications_file)
        else:
            df_class = pd.read_csv(self.classifications_file, header=None)

        df_class = df_class.set_axis(["Classes"], axis=1)

        if len(df_class) == len(self.input_df):
            self.df_feat_class = pd.concat([df_class, self.input_df], axis=1)
            return print("All good.")
        else:
            print(f"Classifications file length: len(df_class)")
            print(f"PyContact file length: len(self.input_df)")
            raise Exception(
                f"""The length of your classifications file ({len(df_class)})
                doesn't match the length of your features file ({len(self.input_df)}).
                If the difference is 1, check if you set the 'header_present' keyword correctly."""
            )

    def filter_features_by_occupancy(self, min_occupancy):
        """
        Filter features such that only features with %occupancy >= the min_occupancy are kept.
        (%occupancy is the % of frames that have a non-zero interaction value).
        In the supervised form %occupancy is determined for each class, meaning only
        observations from 1 class has to meet the cut-off to keep the feature.

        Parameters
        ----------
        min_occupancy : int
            Minimun %occupancy that a feature must have to be retained.

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

    def filter_features_by_type(self, interaction_types_included):
        """Filter features/interactions to use by their type (e.g. hbond or vdws...)"""
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

    def filter_features_by_main_or_side_chain(self, main_side_chain_types_included):
        """Filter features/interactions to only certain combinations of main and side chain."""
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

    def filter_features_by_avg_strength(self, average_strength_cut_off):
        """Filter features/interactions to use by their average strength."""
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
