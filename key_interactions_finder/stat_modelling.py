"""
Calculates differences in the probability distributions of each feature
for the different classes.
This is only available to supervised datasets (i.e. data must has class labels).

2 Classes for end user usage:

1. ClassificationStatModel
    For analysis when the target data is categorical (classification).
    Can only use two classes at a time (i.e., binary classification)

2. RegressionStatModel
    For analysis when the target data is continuous (regression).

These classes both inherit from a parent class called "_ProteinStatModel" which abstracts
as much as their shared behavior as possible.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from key_interactions_finder.utils import _filter_features_by_strings, _prep_out_dir


@dataclass
class _ProteinStatModel:
    """
    Generic class to unify the construction of the classiciation and regression
    classes used by the user. There is no __post_init__ method called by this class
    because each inheriting class has its own instead.

    Attributes
    ----------

    dataset : pd.DataFrame
        Input dataframe.

    out_dir : str
        Directory path to store results files to.
        Default = ""

    interaction_types_included : list, optional
        What types of molecular interactions to generate the correlation matrix for.
        options are one or more of: ["Hbond", "Hydrophobic", "Saltbr", "Other"]
        Default is to include all 4 types.

    scaled_dataset : pd.DataFrame
        Input dataset with all features scaled.

    feature_list : list
        List of all feature labels in the dataset.

    mutual_infos : dict
        Dictionary with each feature's (keys) and mutual informations (values).
        Dictionary is sorted from largest mutual information to smallest.

    Methods
    -------

    _gen_kdes(input_features, kde_bandwidth)
        Generates kernel density estimations for each feature for a single class.

    _scale_features()
        Scale features with MinMaxScaler so that they are all between 0 and 1.

    _per_feature_scores_to_file(per_feat_values, out_file)
        Write the per feature scores to a file.
    """

    # Generated at initialization.
    dataset: pd.DataFrame
    out_dir: str = ""
    interaction_types_included: list = field(default_factory=["Hbond", "Hydrophobic", "Saltbr", "Other"])

    # Generated later.
    scaled_dataset: pd.DataFrame = field(init=False)
    feature_list: list = field(init=False)
    mutual_infos: dict = field(init=False)

    def _gen_kdes(self, input_features: pd.DataFrame, kde_bandwidth: float = 0.02) -> Tuple[np.ndarray, dict]:
        """
        Generates kernel density estimations (kdes) for each feature for a single class.

        Input feature sets are pre-normalised to ranges between 0 and 1,
        allowing for the approximation that using the same bandwidth in the
        kernel density estimation is okay.

        Parameters
        ----------

        input_features : pd.DataFrame
            A dataframe containing all features but with observations for only a single class.
            This can be used to calculate a probabilty distribtion for each feature for each class.

        kde_bandwidth : float
            Bandwidth used to generate the probabilty distribtions for each feature set.
            Note that features are all scaled to be between 0 and 1 before this step.
            Default = 0.02

        Returns
        ----------

        x_values : np.ndarray
            x values for probabilities over the range 0 to 1. Spacing consistent with
            the kernel density estimations generated in this function.

        kdes : dict
            Keys are features and values are an array of kernel density estimations (kdes).
        """
        x_values = np.asarray([value for value in np.arange(0.0, 1.0, kde_bandwidth)])
        x_values = x_values.reshape((int(1 / kde_bandwidth)), 1)

        kdes = {}

        for feature in self.feature_list:
            model = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian")

            feature_values = ((input_features[feature]).to_numpy()).reshape(-1, 1)

            model.fit(feature_values)

            probablities = np.exp(model.score_samples(x_values))
            kdes[feature] = probablities

        return x_values, kdes

    def _scale_features(self) -> pd.DataFrame:
        """
        Scale features with MinMaxScaler so that they are all between 0 and 1.
        This is important for the KDE as I am using the same bandwidth parameter throughout.

        Returns
        ----------

        pd.DataFrame
            Dataframe with all features scaled between 0 and 1.
        """
        scaler = MinMaxScaler()
        feature_values = (self.dataset.drop("Target", axis=1)).to_numpy()
        scaler.fit(feature_values)
        feature_values_scaled = scaler.transform(feature_values)

        scaled_dataset = pd.DataFrame.from_records(feature_values_scaled, columns=self.feature_list)
        scaled_dataset.insert(0, "Target", self.dataset["Target"])

        return scaled_dataset

    @staticmethod
    def _per_feature_scores_to_file(per_feat_values: dict, out_file: str) -> None:
        """
        Write the per feature scores to a file.

        Parameters
        ----------

        per_feat_values : dict
            Dictionary of feature names and their scores to write to disk.

        out_file : str
            The full path to write the file too.
        """
        with open(out_file, "w", newline="", encoding="utf-8") as file_out:
            csv_out = csv.writer(file_out)
            csv_out.writerow(["Feature", "Score"])
            csv_out.writerows(per_feat_values.items())
            print(f"{out_file} written to disk.")


@dataclass
class ClassificationStatModel(_ProteinStatModel):
    """
    Handles the generation of statistical models for PyContact data sets
    when the target is made up of two unqiue class labels.

    Note that most attributes are inherited from _ProteinStatModel.

    Attributes
    ----------

    dataset : pd.DataFrame
        Input dataframe.

    class_names : list
        Class labels inside the column "Target" of the dataset to model.
        You can only use two classes for this approach.

    out_dir : str
        Directory path to store results files to.
        Default = ""

    interaction_types_included : list, optional
        What types of molecular interactions to generate the correlation matrix for.
        options are one or more of: ["Hbond", "Hydrophobic", "Saltbr", "Other"]
        Default is to include all 4 types.

    scaled_dataset : pd.DataFrame
        Input dataset with all features scaled.

    feature_list : list
        List of all feature labels in the dataset.

    x_values : np.ndarray
        Values on the x-axis for plotting the kernel density estimations.

    kdes : dict
        Nested dictionary. Outer layer keys are class names, and values
        are a dictionary of each feature (as inner key) and values of a
        nested array of kernel density estimations (kdes).

    js_distances : dict
        Dictionary with each feature's (keys) and Jensen Shannon distance (values).
        Dictionary is sorted from largest Jensen Shannon distance to smallest.

    mutual_infos : dict
        Dictionary with each feature's (keys) and mutual informations (values).
        Dictionary is sorted from largest mutual information to smallest.

    Methods
    -------

    calc_mutual_info_to_target(save_result=True)
        Calculate the mutual information between each feature to the target classes.

    calc_js_distances(kde_bandwidth=0.02, save_result=True)
        Calculate the Jensen-Shannon (JS) distance (metric) between each feature to
        the target classes.
    """

    # Generated at initialization.
    class_names: list = field(default_factory=[])
    # Generated later.
    x_values: np.ndarray = field(init=False)
    kdes: dict = field(init=False)
    js_distances: dict = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self) -> None:
        """Filter, rescale and calc the kdes for each feature."""
        self.out_dir = _prep_out_dir(self.out_dir)

        self.x_values = np.empty(shape=(0, 0))
        self.kdes = {}
        self.js_distances = {}
        self.mutual_infos = {}

        if sorted(self.interaction_types_included) != sorted(["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features_by_strings(
                dataset=self.dataset, strings_to_preserve=self.interaction_types_included
            )

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Target")

        # Features need to be scaled in order to use same bandwidth throughout.
        self.scaled_dataset = self._scale_features()

        if len(self.class_names) != 2:
            raise ValueError(
                "The number of classes to compare should be 2. \n"
                + "Please use a list of 2 items for the parameter: 'class_names'."
            )

    def calc_mutual_info_to_target(self, save_result: bool = True):
        """
        Calculate the mutual information between each feature to the 2 target classes.
        Note that Sklearns implementation (used here) is designed for "raw datasets"
        (i.e., do not feed in a probability distribution, instead feed in the observations).

        Further, the mutual information values calculated from Sklearns implementation are
        scaled by the natural logarithm of 2. In this implementation,
        the results are re-scaled to be linear.

        Parameters
        ----------

        save_result : Optional[bool] = True
            Save result to disk or not.
            Optional, default is to save.
        """
        df_features = self.scaled_dataset.drop("Target", axis=1)
        features_array = df_features.to_numpy()
        classes = self.scaled_dataset["Target"].to_numpy()

        mutual_info_raw = mutual_info_classif(features_array, classes)
        mutual_info_rescaled = np.around((np.exp(mutual_info_raw) - 1), 5)

        self.mutual_infos = dict(zip(df_features.columns, mutual_info_rescaled, strict=True))
        self.mutual_infos = {k: v for k, v in sorted(self.mutual_infos.items(), key=lambda item: item[1], reverse=True)}

        print("Mutual information scores calculated.")

        if save_result:
            out_file_path = Path(self.out_dir, "Mutual_Information_Per_Feature_Scores.csv")
            self._per_feature_scores_to_file(per_feat_values=self.mutual_infos, out_file=out_file_path)
            print("You can also access these results via the class attribute: 'mutual_infos'.")

    def calc_js_distances(self, kde_bandwidth: float = 0.02, save_result: bool = True):
        """
        Calculate the Jensen-Shannon (JS) distance (metric) between each feature to
        the target classes.
        Requires that each feature is described by a probabilty distribution.

        Parameters
        ----------

        kde_bandwidth : Optional[float]
            Bandwidth used to generate the probabilty distribtions for each feature set.
            Note that features are all scaled to be between 0 and 1 before this step.
            Optional, default = 0.02

        save_result : Optional[bool] = True
            Save result to disk or not.
            Optional, default is to save.
        """
        for class_name in self.class_names:
            # split observations into each class first.
            per_class_dataset = self.scaled_dataset[(self.scaled_dataset["Target"] == class_name)]

            # generate kdes for each class.
            self.x_values, self.kdes[class_name] = self._gen_kdes(
                input_features=per_class_dataset, kde_bandwidth=kde_bandwidth
            )

        # iterate through each feature and calc js dist.
        for feature in self.feature_list:
            distrib_1 = self.kdes[self.class_names[0]][feature]
            distrib_2 = self.kdes[self.class_names[1]][feature]

            js_dist = np.around(jensenshannon(distrib_1, distrib_2, base=2), 5)

            self.js_distances.update({feature: js_dist})

        self.js_distances = {k: v for k, v in sorted(self.js_distances.items(), key=lambda item: item[1], reverse=True)}

        print("Jensen-Shannon (JS) distances calculated.")

        if save_result:
            out_file_path = Path(self.out_dir, "Jensen_Shannon_Per_Feature_Scores.csv")
            self._per_feature_scores_to_file(per_feat_values=self.js_distances, out_file=out_file_path)

            print("You can also access these results via the class attribute: 'js_distances'.")


@dataclass
class RegressionStatModel(_ProteinStatModel):
    """
    Handles the generation of statistical models for PyContact data sets
    when the target variable is continous.

    Note that several attributes listed below are inherited from _ProteinStatModel.

    Attributes
    ----------

    dataset : pd.DataFrame
        Input dataframe.

    out_dir : str
        Directory path to store results files to.
        Default = ""

    interaction_types_included : list, optional
        What types of molecular interactions to generate the correlation matrix for.
        options are one or more of: ["Hbond", "Hydrophobic", "Saltbr", "Other"]
        Default is to include all 4 types.

    scaled_dataset : pd.DataFrame
        Input dataset with all features scaled.

    feature_list : list
        List of all feature labels in the dataset.

    mutual_infos : dict
        Dictionary with each feature's (keys) and mutual informations (values).
        Dictionary is sorted from largest mutual information to smallest.

    linear_correlations : dict
        Dictionary with each feature's (keys) and linear correlations (values).
        Dictionary is sorted from largest (absolute) linear correlation to smallest.

    Methods
    -------

    calc_mutual_info_to_target(save_result=True)
        Calculate the mutual information between each feature and the target.

    calc_linear_correl_to_target(save_result=True)
        Calculate the pearson correlation coeffcient between each feature and the target.
    """

    # Attribute is generated after initiziliation.
    linear_correlations: dict = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self) -> None:
        """Filter and rescale the features."""
        self.out_dir = _prep_out_dir(self.out_dir)

        self.mutual_infos = {}
        self.linear_correlations = {}

        if sorted(self.interaction_types_included) != sorted(["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features_by_strings(
                dataset=self.dataset, strings_to_preserve=self.interaction_types_included
            )

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Target")

        # Features need to be scaled in order to use same bandwidth throughout.
        self.scaled_dataset = self._scale_features()

    def calc_mutual_info_to_target(self, save_result: bool = True) -> None:
        """
        Calculate the mutual information between each feature and the target.
        The target variable should be continuous.
        Note that Sklearns implementation (used here) is designed for "raw datasets"
        (i.e., do not feed in a probability distribution, instead feed in the observations).

        Further, the mutual information values calculated from Sklearns implementation are
        scaled by the natural logarithm of 2. In this implementation,
        the results are re-scaled to be linear.

        Parameters
        ----------

        save_result : Optional[bool] = True
            Save result to disk or not.
            Optional, default is to save.
        """
        df_features = self.scaled_dataset.drop("Target", axis=1)
        features_array = df_features.to_numpy()
        target_values = self.scaled_dataset["Target"].to_numpy()

        mutual_info_raw = mutual_info_regression(features_array, target_values)
        mutual_info_rescaled = np.around((np.exp(mutual_info_raw) - 1), 5)

        self.mutual_infos = dict(zip(df_features.columns, mutual_info_rescaled, strict=True))
        self.mutual_infos = {k: v for k, v in sorted(self.mutual_infos.items(), key=lambda item: item[1], reverse=True)}

        print("Mutual information scores calculated.")

        if save_result:
            out_file_path = Path(self.out_dir, "Mutual_Information_Per_Feature_Scores.csv")
            self._per_feature_scores_to_file(per_feat_values=self.mutual_infos, out_file=out_file_path)

            print("You can also access these results via the class attribute: 'mutual_infos'.")

    def calc_linear_correl_to_target(self, save_result: bool = True) -> None:
        """
        Calculate the pearson correlation coeffcient between each feature and the target.

        Parameters
        ----------

        save_result : Optional[bool] = True
            Save result to disk or not.
            Optional, default is to save.
        """
        target = self.dataset["Target"]
        features = self.dataset.drop(["Target"], axis=1)

        correlations = features.corrwith(target).to_frame()
        sorted_correlations = correlations.sort_values(by=0, key=abs, ascending=False)
        self.linear_correlations = sorted_correlations.to_dict(orient="dict")[0]

        print("Linear correlations calculated.")

        if save_result:
            out_file_path = Path(self.out_dir, "Linear_Correlations_Per_Feature_Scores.csv")
            self._per_feature_scores_to_file(per_feat_values=self.linear_correlations, out_file=out_file_path)

            print("You can also access these results via the class attribute: 'linear_correlations'.")
