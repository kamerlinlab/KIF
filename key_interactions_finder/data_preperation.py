"""
DocString here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


class FeatureData(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised feature datasets."""

    @abstractmethod
    def filter_features(self, df, min_occupancy):
        """Filter features/interactions to use with a minimum allowed observation cut-off."""


@dataclass
class UnsupervisedFeautureData(FeatureData):
    """FeatureData Class without any classification data."""
    scaling_method: str = "minmax"  # If I put this here, I should be passing it when I make the class??

    def filter_features(self, df, min_occupancy):
        """
        Filter features such that only features with %occupancy >= the min_occupancy are included.
        (%occupancy is the percantage of frames that have a non-zero interaction value for a given feature).
        The supervised form has a different implementation - hence the seperation.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas Dataframe containing features ready to be filtered.

        min_occupancy : int
            Minimun percantage occupancy that a feature must have to be retained.

        Returns
        -------
        df_filtered : pd.DataFrame
            Filtered dataframe.

        """
        return df.loc[:, (df != 0).mean() > (min_occupancy/100)]


@dataclass
class SupervisedFeatureData(FeatureData):
    """FeatureData Class with classification data included."""

    def add_clasifications(self, df, classifications_file, header_present=True):
        """Merge per frame classification results to dataframe."""
        if header_present:
            df_class = pd.read_csv(classifications_file)
        else:
            df_class = pd.read_csv(classifications_file, header=None)

        df_class = df_class.set_axis(["Classes"], axis=1)
        df_feat_class = pd.concat([df_class, df], axis=1)
        return df_feat_class

    def filter_features(self, df, min_occupancy):
        """
        Filter features such that only features with %occupancy >= the min_occupancy are included.
        (%occupancy is the percantage of frames that have a non-zero interaction value for a given feature).
        In the supervised form %occupancy is determined for each class (so only 1 class has to meet the cut-off).

        Parameters
        ----------
        df : pd.DataFrame
            Pandas Dataframe containing features ready to be filtered (and also includes a classifications column).

        min_occupancy : int
            Minimun percantage occupancy that a feature must have to be retained.

        Returns
        -------
        df_filtered : pd.DataFrame
            Filtered dataframe.

        """
        keep_cols = ["Classes"]  # always want "Classes" present...
        # add any feature that meets the cutoff to the "keep_cols" list.
        for class_label in list(df["Classes"].unique()):
            df_single_class = df[(df["Classes"] == class_label)]
            keep_cols_single_class = list(
                (df_single_class.loc[:, (df_single_class !=
                                         0).mean() > (min_occupancy/100)]).columns
            )
            keep_cols.extend(keep_cols_single_class)

        df_filtered = df[list(set(keep_cols))]
        return df_filtered
