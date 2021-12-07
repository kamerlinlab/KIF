"""
Models the distributions of each feature for the different classes.
This is only available to datasets with labels.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import jensenshannon

from key_interactions_finder.utils import _prep_out_dir, _filter_features


@dataclass
class ProteinStatModel():
    """Descript. """

    # Generated at runtime.
    dataset: pd.core.frame.DataFrame
    class_names: list
    out_dir: str = ""
    kde_bandwidth: float = 0.02
    interaction_types_included: list = field(
        default_factory=["Hbond", "Hydrophobic", "Saltbr", "Other"])

    # Generated later.

    per_class_datasets: dict = field(init=False)
    feature_list: list = field(init=False)

    x_values: np.ndarray = field(init=False)
    probablity_distributions: dict = field(init=False)
    js_distances: dict = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Descript. """
        self.out_dir = _prep_out_dir(self.out_dir)

        if sorted(self.interaction_types_included) != sorted(["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features(
                self.dataset, self.interaction_types_included)

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Classes")

        # Features need to be scaled in order to use same bandwidth throughout.
        scaled_dataset = self._scale_features()
        print(scaled_dataset)

        if len(self.class_names) != 2:
            raise ValueError(
                "The number of classes to compare should be 2. \n" +
                "Please use a list of 2 items for the parameter: 'class_names'.")

        self.per_class_datasets = {}
        for class_name in self.class_names:
            self.per_class_datasets[class_name] = scaled_dataset[(
                scaled_dataset["Classes"] == class_name)]

        self.x_values, self.probablity_distributions = self._gen_prob_distributions()

    def calc_mutual_info_scores(self):
        """
        Calculate the mutual information between each feature to the target classes.
        Requires that each feature is described by a probabilty distrubtion.

        Returns
        ----------
        describe
            desribe
        """

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

            js_dist = np.around(jensenshannon(distrub_1, distrub_2, base=2), 8)

            self.js_distances.update({feature: js_dist})
            self.js_distances = {k: v for k, v in sorted(
                self.js_distances.items(), key=lambda item: item[1], reverse=True)}

        return self.js_distances

    def _gen_prob_distributions(self):
        """
        From each feature generate a probability distrubtion for each class.

        Input feature sets are pre-normalised to ranges between 0 and 1,
        allowing for the approximation that a consistent bandwith will be okay.

        Returns
        ----------
        x_values : np.array
            x values for probabilities over the range 0 to 1. Spacing consistent with
            the probablity distributions generated in this function.

        probablity_distributions : dict
            Nested dictionary. Outer layer keys are class names, and values
            are a dictionary of each feature (as inner key) and values of a
            nested array of probabilities. TODO.
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
        Scale features with MinMaxScaler so that they are all between 1 and 0.
        This is important as for the KDE, I am using the same bandwidth parameter throughout.

        TODO - UPDATE DOCs.

        """
        scaler = MinMaxScaler()
        feature_values = (self.dataset.drop("Classes", axis=1)).to_numpy()
        scaler.fit(feature_values)
        feature_values_scaled = scaler.transform(feature_values)

        print(feature_values_scaled)

        # rebuild the df
        scaled_dataset = pd.DataFrame.from_records(
            feature_values_scaled, columns=self.feature_list)
        scaled_dataset.insert(0, "Classes", self.dataset["Classes"])

        return scaled_dataset
