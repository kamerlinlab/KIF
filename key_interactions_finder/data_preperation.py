"""
Add docstring here.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


class FeatureData(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised feature datasets."""

    @abstractmethod
    def filter_features(self, df, min_occupancy):
        """Filter features/interactions to use with a minimum allowed observation cut-off."""


@dataclass
class UnsupervisedFeautureData(FeatureData):
    """FeatureData Class without any classification data."""

    input_df: pd.core.frame.DataFrame
    df_filtered: pd.core.frame.DataFrame = field(init=False)

    def filter_features(self, min_occupancy):
        """
        Filter features such that only features with %occupancy >= the min_occupancy are included.
        (%occupancy is the percantage of frames that have a non-zero interaction value for a given feature).
        The supervised form has a different implementation - hence the seperation.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            Pandas Dataframe containing features ready to be filtered.

        min_occupancy : int
            Minimun percantage occupancy that a feature must have to be retained.

        Returns
        -------
        pd.core.frame.DataFrame
            Filtered dataframe.

        """
        self.df_filtered = self.input_df.loc[:, (
            self.input_df != 0).mean() > (min_occupancy/100)]
        return self.df_filtered


@dataclass
class SupervisedFeatureData(FeatureData):
    """FeatureData Class with classification data included."""

    input_df: pd.core.frame.DataFrame
    classifications_file: str
    header_present: bool = True

    df_feat_class: pd.core.frame.DataFrame = field(init=False)
    df_filtered: pd.core.frame.DataFrame = field(init=False)

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
                doesn't match the length of your features file ({len(self.input_df)})!
                If the difference is 1, check if you set the 'header_present' keyword correctly."""
            )

    def filter_features(self, min_occupancy):
        """
        Filter features such that only features with %occupancy >= the min_occupancy are kept.
        (%occupancy is the % of frames that have a non-zero interaction value).
        In the supervised form %occupancy is determined for each class,
        meaning only observation from 1 class has to meet the cut-off.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            df containing PyContact features to be filtered and a column with classes.

        min_occupancy : int
            Minimun %occupancy that a feature must have to be retained.

        Returns
        -------
        pd.core.frame.DataFrame
            Filtered dataframe.

        """
        keep_cols = ["Classes"]  # always want "Classes" present...

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
