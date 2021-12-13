"""
Models the distributions of each feature for the different classes.
This is only available to datasets with labels.
"""

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
    """Handles the generation of stastical models for PyContact feature sets."""

    # Generated at initialization.
    dataset: pd.core.frame.DataFrame
    class_names: list = field(default_factory=[])
    out_dir: str = ""
    kde_bandwidth: float = 0.02
    interaction_types_included: list = field(
        default_factory=["Hbond", "Hydrophobic", "Saltbr", "Other"])

    # Generated later.
    scaled_dataset: pd.core.frame.DataFrame = field(init=False)
    per_class_datasets: dict = field(init=False)
    feature_list: list = field(init=False)
    x_values: np.ndarray = field(init=False)
    probablity_distributions: dict = field(init=False)
    js_distances: dict = field(init=False)
    mutual_infos: dict = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Descript. """
        self.out_dir = _prep_out_dir(self.out_dir)

        if sorted(self.interaction_types_included) != sorted(
            ["Hbond", "Hydrophobic", "Saltbr", "Other"]
        ):
            self.dataset = _filter_features_by_strings(
                dataset=self.dataset,
                strings_to_preserve=self.interaction_types_included
            )

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Classes")

        # Features need to be scaled in order to use same bandwidth throughout.
        self.scaled_dataset = self._scale_features()
        print(self.scaled_dataset)

        if len(self.class_names) != 2:
            raise ValueError(
                "The number of classes to compare should be 2. \n" +
                "Please use a list of 2 items for the parameter: 'class_names'.")

        self.per_class_datasets = {}
        for class_name in self.class_names:
            self.per_class_datasets[class_name] = self.scaled_dataset[(
                self.scaled_dataset["Classes"] == class_name)]

        self.x_values, self.probablity_distributions = self._gen_prob_distributions()

    def calc_mutual_info_to_target(self):
        """
        Calculate the mutual information between each feature to the target classes.
        Note that Sklearns implementation (used here) is designed for "raw datasets"
        (i.e., do not feed in a probability distribution, instead feed in the observations).

        Returns
        ----------
        pd.core.frame.DataFrame
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

    def calc_js_distances(self):
        """
        Calculate the Jensen-Shannon (JS) distance (metric) between each feature to
        the target classes.
        Requires that each feature is described by a probabilty distrubtion.

        Returns
        ----------
        dict
            Dictionary with each feature's (keys) JS distance (values) that is
            sorted from largest JS distance to smallest.
        """
        self.js_distances = {}
        for feature in self.feature_list:
            distrub_1 = self.probablity_distributions[self.class_names[0]][feature]
            distrub_2 = self.probablity_distributions[self.class_names[1]][feature]

            js_dist = np.around(jensenshannon(distrub_1, distrub_2, base=2), 5)

            self.js_distances.update({feature: js_dist})

        self.js_distances = {k: v for k, v in sorted(
            self.js_distances.items(), key=lambda item: item[1], reverse=True)}

        return self.js_distances

    def _gen_prob_distributions(self):
        """
        For each feature generate a probability distrubtion for each class.

        Input feature sets are pre-normalised to ranges between 0 and 1,
        allowing for the approximation that using the same bandwith in the
        kernel density estimation is okay.

        Returns
        ----------
        x_values : np.array
            x values for probabilities over the range 0 to 1. Spacing consistent with
            the probablity distributions generated in this function.

        probablity_distributions : dict
            Nested dictionary. Outer layer keys are class names, and values
            are a dictionary of each feature (as inner key) and values of a
            nested array of probabilities.
        """
        x_values = np.asarray(
            [value for value in np.arange(0.0, 1.0, self.kde_bandwidth)])
        x_values = x_values.reshape((int(1/self.kde_bandwidth)), 1)

        probablity_distributions = {}

        for class_name in self.class_names:
            per_feature_probs = {}
            for feature in self.feature_list:
                model = KernelDensity(
                    bandwidth=self.kde_bandwidth, kernel="gaussian")

                feature_values = (
                    (self.per_class_datasets[class_name][feature]).to_numpy()).reshape(-1, 1)
                model.fit(feature_values)

                probablities = np.exp(model.score_samples(x_values))
                per_feature_probs[feature] = probablities

            probablity_distributions[class_name] = per_feature_probs

        return x_values, probablity_distributions

    def _scale_features(self):
        """
        Scale features with MinMaxScaler so that they are all between 0 and 1.
        This is important for the KDE as I am using the same bandwidth parameter throughout.

        Returns
        ----------
        pd.core.frame.DataFrame
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
