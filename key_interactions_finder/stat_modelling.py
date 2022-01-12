"""
Calculates differences in the probabilty distributions of each feature
for the different classes.
This is only available to supervised datasets (i.e. data must has class labels).
"""
from typing import Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import jensenshannon

from key_interactions_finder.utils import _prep_out_dir, _filter_features_by_strings


@dataclass
class ProteinStatModel():
    """
    Handles the generation of stastical models for PyContact data sets.
    Requires dataset to contain two unqiue class labels.

    Attributes
    ----------
    dataset : pd.DataFrame
        Input dataframe.

    class_names : list
        Class labels inside the column "Classes" of the dataset to model.
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
        Values on the x-axis for plotting the probability distrubitons.

    probablity_distributions : dict
        Nested dictionary. Outer layer keys are class names, and values
        are a dictionary of each feature (as inner key) and values of a
        nested array of probabilities.

    js_distances : dict
        Dictionary with each feature's (keys) and Jensen Shannon distance (values).
        Dictionary is sorted from largest Jensen Shannon distance to smallest.

    mutual_infos : dict
        Dictionary with each feature's (keys) and mutual informations (values).
        Dictionary is sorted from largest mutual information to smallest.

    Methods
    -------
    calc_mutual_info_to_target()
        Calculate the mutual information between each feature to the target classes.

    calc_js_distances(kde_bandwidth=0.02)
        Calculate the Jensen-Shannon (JS) distance (metric) between each feature to
        the target classes.
    """

    # Generated at initialization.
    dataset: pd.DataFrame
    class_names: list = field(default_factory=[])
    out_dir: str = ""
    interaction_types_included: list = field(
        default_factory=["Hbond", "Hydrophobic", "Saltbr", "Other"])

    # Generated later.
    scaled_dataset: pd.DataFrame = field(init=False)
    feature_list: list = field(init=False)
    x_values: np.ndarray = field(init=False)
    probablity_distributions: dict = field(init=False)
    js_distances: dict = field(init=False)
    mutual_infos: dict = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self) -> None:
        """Filters, rescales and generates the probability distributions for each feature."""
        self.js_distances = {}
        self.mutual_infos = {}

        self.out_dir = _prep_out_dir(self.out_dir)

        if sorted(self.interaction_types_included) != sorted(
                ["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features_by_strings(
                dataset=self.dataset,
                strings_to_preserve=self.interaction_types_included
            )

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Classes")

        # Features need to be scaled in order to use same bandwidth throughout.
        self.scaled_dataset = self._scale_features()

        if len(self.class_names) != 2:
            raise ValueError(
                "The number of classes to compare should be 2. \n" +
                "Please use a list of 2 items for the parameter: 'class_names'.")

    def calc_mutual_info_to_target(self) -> pd.DataFrame:
        """
        Calculate the mutual information between each feature to the target classes.
        Note that Sklearns implementation (used here) is designed for "raw datasets"
        (i.e., do not feed in a probability distribution, instead feed in the observations).

        Returns
        ----------
        pd.DataFrame
            Dataframe containing all features and the mutual information scores.
            Mutual information units are "nits".
        """
        df_features = self.scaled_dataset.drop("Classes", axis=1)
        features_array = df_features.to_numpy()
        classes = self.scaled_dataset["Classes"].to_numpy()

        mutual_info_raw = np.around(
            mutual_info_classif(features_array, classes), 5)
        self.mutual_infos = dict(zip(df_features.columns, mutual_info_raw))
        self.mutual_infos = {k: v for k, v in sorted(
            self.mutual_infos.items(), key=lambda item: item[1], reverse=True)}

        return self.mutual_infos

    def calc_js_distances(self, kde_bandwidth: float = 0.02) -> dict:
        """
        Calculate the Jensen-Shannon (JS) distance (metric) between each feature to
        the target classes.
        Requires that each feature is described by a probabilty distribution.

        Parameters
        ----------
        kde_bandwidth : float
            Bandwidth used to generate the probabilty distribtions for each feature set.
            Note that features are all scaled to be between 0 and 1 before this step.
            Default = 0.02

        Returns
        ----------
        dict
            Dictionary with each feature's (keys) JS distance (values).
            Dictionary is sorted from largest JS distance to smallest.
        """

        self.x_values, self.probablity_distributions = self._gen_prob_distributions(
            kde_bandwidth=kde_bandwidth)

        for feature in self.feature_list:
            distrib_1 = self.probablity_distributions[self.class_names[0]][feature]
            distrib_2 = self.probablity_distributions[self.class_names[1]][feature]

            js_dist = np.around(jensenshannon(distrib_1, distrib_2, base=2), 5)

            self.js_distances.update({feature: js_dist})

        self.js_distances = {k: v for k, v in sorted(
            self.js_distances.items(), key=lambda item: item[1], reverse=True)}

        return self.js_distances

    def _gen_prob_distributions(self, kde_bandwidth: float) -> Tuple[np.ndarray, dict]:
        """
        For each feature generate a probability distributions of it scores
        towards each class.

        Input feature sets are pre-normalised to ranges between 0 and 1,
        allowing for the approximation that using the same bandwith in the
        kernel density estimation is okay.

        Parameters
        ----------
        kde_bandwidth : float
            Bandwidth used to generate the probabilty distribtions for each feature set.
            Note that features are all scaled to be between 0 and 1 before this step.
            Default = 0.02

        Returns
        ----------
        x_values : np.ndarray
            x values for probabilities over the range 0 to 1. Spacing consistent with
            the probablity distributions generated in this function.

        probablity_distributions : dict
            Nested dictionary. Outer layer keys are class names, and values
            are a dictionary of each feature (as inner key) and values of a
            nested array of probabilities.
        """
        per_class_datasets = {}
        for class_name in self.class_names:
            per_class_datasets[class_name] = self.scaled_dataset[(
                self.scaled_dataset["Classes"] == class_name)]

        x_values = np.asarray(
            [value for value in np.arange(0.0, 1.0, kde_bandwidth)])
        x_values = x_values.reshape((int(1/kde_bandwidth)), 1)

        probablity_distributions = {}
        for class_name in self.class_names:
            per_feature_probs = {}
            for feature in self.feature_list:
                model = KernelDensity(
                    bandwidth=kde_bandwidth, kernel="gaussian")

                feature_values = (
                    (per_class_datasets[class_name][feature]).to_numpy()).reshape(-1, 1)
                model.fit(feature_values)

                probablities = np.exp(model.score_samples(x_values))
                per_feature_probs[feature] = probablities

            probablity_distributions[class_name] = per_feature_probs

        return x_values, probablity_distributions

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
        feature_values = (self.dataset.drop("Classes", axis=1)).to_numpy()
        scaler.fit(feature_values)
        feature_values_scaled = scaler.transform(feature_values)

        scaled_dataset = pd.DataFrame.from_records(
            feature_values_scaled, columns=self.feature_list)
        scaled_dataset.insert(0, "Classes", self.dataset["Classes"])

        return scaled_dataset
