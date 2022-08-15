"""
Calculates differences in the probabilty distributions of each feature
for the different classes.
This is only available to supervised datasets (i.e. data must has class labels).
"""
from typing import Tuple
from dataclasses import dataclass, field
import csv
import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.spatial.distance import jensenshannon

from key_interactions_finder.utils import _prep_out_dir, _filter_features_by_strings


@dataclass
class _ProteinStatModel():
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
    _gen_prob_distributions(input_features, kde_bandwidth)
        Generates probability distributions for each feature for a single class.

    _scale_features()
        Scale features with MinMaxScaler so that they are all between 0 and 1.

    _per_feature_importances_to_file(per_feat_values, out_file)
        Write out the per feature importances to a file.
    """
    # Generated at initialization.
    dataset: pd.DataFrame
    out_dir: str = ""
    interaction_types_included: list = field(
        default_factory=["Hbond", "Hydrophobic", "Saltbr", "Other"])

    # Generated later.
    scaled_dataset: pd.DataFrame = field(init=False)
    feature_list: list = field(init=False)
    mutual_infos: dict = field(init=False)

    def _gen_prob_distributions(self, input_features: pd.DataFrame, kde_bandwidth: float) -> Tuple[np.ndarray, dict]:
        """
        Generates probability distributions for each feature for a single class.

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
            the probability distributions generated in this function.

        probability_distributions : dict
            Keys are features and values are an array of probabilities.
        """
        x_values = np.asarray(
            [value for value in np.arange(0.0, 1.0, kde_bandwidth)])
        x_values = x_values.reshape((int(1/kde_bandwidth)), 1)

        probability_distributions = {}

        for feature in self.feature_list:
            model = KernelDensity(
                bandwidth=kde_bandwidth, kernel="gaussian")

            feature_values = (
                (input_features[feature]).to_numpy()).reshape(-1, 1)

            model.fit(feature_values)

            probablities = np.exp(model.score_samples(x_values))
            probability_distributions[feature] = probablities

        return x_values, probability_distributions

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

        scaled_dataset = pd.DataFrame.from_records(
            feature_values_scaled, columns=self.feature_list)
        scaled_dataset.insert(0, "Target", self.dataset["Target"])

        return scaled_dataset

    @staticmethod
    def _per_feature_importances_to_file(per_feat_values: dict, out_file: str) -> None:
        """
        Write out the per feature importances to a file.

        Parameters
        ----------
        per_feat_values : dict
            Dictionary of feature names and their scores to write to disk.

        out_file : str
            The full path to write the file too.
        """
        with open(out_file, "w", newline="") as file_out:
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
        Values on the x-axis for plotting the probability distrubitons.

    probability_distributions : dict
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
    class_names: list = field(default_factory=[])
    # Generated later.
    x_values: np.ndarray = field(init=False)
    probability_distributions: dict = field(init=False)
    js_distances: dict = field(init=False)

    # Called at the end of the dataclass's initialization procedure.
    def __post_init__(self) -> None:
        """Filter, rescale and calc the probability distributions for each feature."""
        self.out_dir = _prep_out_dir(self.out_dir)

        self.x_values = np.empty(shape=(0, 0))
        self.probability_distributions = {}
        self.js_distances = {}
        self.mutual_infos = {}

        if sorted(self.interaction_types_included) != sorted(
                ["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features_by_strings(
                dataset=self.dataset,
                strings_to_preserve=self.interaction_types_included
            )

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Target")

        # Features need to be scaled in order to use same bandwidth throughout.
        self.scaled_dataset = self._scale_features()

        if len(self.class_names) != 2:
            raise ValueError(
                "The number of classes to compare should be 2. \n" +
                "Please use a list of 2 items for the parameter: 'class_names'.")

    def calc_mutual_info_to_target(self):
        """
        Calculate the mutual information between each feature to the 2 target classes.
        Note that Sklearns implementation (used here) is designed for "raw datasets"
        (i.e., do not feed in a probability distribution, instead feed in the observations).

        Further, the mutual information values calculated from Sklearns implementation are
        scaled by the natural logarithm of 2. In this implementation,
        the results are re-scaled to be linear.
        """
        df_features = self.scaled_dataset.drop("Target", axis=1)
        features_array = df_features.to_numpy()
        classes = self.scaled_dataset["Target"].to_numpy()

        mutual_info_raw = mutual_info_classif(features_array, classes)
        mutual_info_rescaled = np.around((np.exp(mutual_info_raw) - 1), 5)

        self.mutual_infos = dict(
            zip(df_features.columns, mutual_info_rescaled))
        self.mutual_infos = {k: v for k, v in sorted(
            self.mutual_infos.items(), key=lambda item: item[1], reverse=True)}

        print("Mutual information scores calculated.")

        out_file = self.out_dir + "Mutual_Information_Per_Feature_Scores.csv"
        self._per_feature_importances_to_file(
            per_feat_values=self.mutual_infos,
            out_file=out_file
        )
        print("You can also access these results via the class attribute: 'mutual_infos'.")

    def calc_js_distances(self, kde_bandwidth: float = 0.02):
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
        """
        for class_name in self.class_names:
            # split observations into each class first.
            per_class_dataset = self.scaled_dataset[(
                self.scaled_dataset["Target"] == class_name)]

            # generate probability distr. for each class.
            self.x_values, self.probability_distributions[class_name] = self._gen_prob_distributions(
                input_features=per_class_dataset,
                kde_bandwidth=kde_bandwidth)

        # iterate through each feature and calc js dist.
        for feature in self.feature_list:
            distrib_1 = self.probability_distributions[self.class_names[0]][feature]
            distrib_2 = self.probability_distributions[self.class_names[1]][feature]

            js_dist = np.around(jensenshannon(distrib_1, distrib_2, base=2), 5)

            self.js_distances.update({feature: js_dist})

        self.js_distances = {k: v for k, v in sorted(
            self.js_distances.items(), key=lambda item: item[1], reverse=True)}

        print("Jensen-Shannon (JS) distances calculated.")

        out_file = self.out_dir + "Jensen_Shannon_Per_Feature_Scores.csv"
        self._per_feature_importances_to_file(
            per_feat_values=self.js_distances,
            out_file=out_file
        )
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
    calc_mutual_info_to_target()
        Calculate the mutual information between each feature and the target.

    calc_linear_correl_to_target()
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

        if sorted(self.interaction_types_included) != sorted(
                ["Hbond", "Hydrophobic", "Saltbr", "Other"]):
            self.dataset = _filter_features_by_strings(
                dataset=self.dataset,
                strings_to_preserve=self.interaction_types_included
            )

        self.feature_list = list(self.dataset.columns)
        self.feature_list.remove("Target")

        # Features need to be scaled in order to use same bandwidth throughout.
        self.scaled_dataset = self._scale_features()

    def calc_mutual_info_to_target(self) -> None:
        """
        Calculate the mutual information between each feature and the target.
        The target variable should be continuous.
        Note that Sklearns implementation (used here) is designed for "raw datasets"
        (i.e., do not feed in a probability distribution, instead feed in the observations).

        Further, the mutual information values calculated from Sklearns implementation are
        scaled by the natural logarithm of 2. In this implementation,
        the results are re-scaled to be linear.
        """
        df_features = self.scaled_dataset.drop("Target", axis=1)
        features_array = df_features.to_numpy()
        target_values = self.scaled_dataset["Target"].to_numpy()

        mutual_info_raw = mutual_info_regression(features_array, target_values)
        mutual_info_rescaled = np.around((np.exp(mutual_info_raw) - 1), 5)

        self.mutual_infos = dict(
            zip(df_features.columns, mutual_info_rescaled))
        self.mutual_infos = {k: v for k, v in sorted(
            self.mutual_infos.items(), key=lambda item: item[1], reverse=True)}

        print("Mutual information scores calculated.")

        out_file = self.out_dir + "Mutual_Information_Per_Feature_Scores.csv"
        self._per_feature_importances_to_file(
            per_feat_values=self.mutual_infos,
            out_file=out_file
        )
        print("You can also access these results via the class attribute: 'mutual_infos'.")

    def calc_linear_correl_to_target(self) -> None:
        """Calculate the pearson correlation coeffcient between each feature and the target."""
        target = self.dataset["Target"]
        features = self.dataset.drop(["Target"], axis=1)

        correlations = features.corrwith(target).to_frame()
        sorted_correlations = correlations.sort_values(
            by=0, key=abs, ascending=False)
        self.linear_correlations = sorted_correlations.to_dict(
            orient='dict')[0]

        print("Linear correlations calculated.")

        out_file = self.out_dir + "Linear_Correlations_Per_Feature_Scores.csv"
        self._per_feature_importances_to_file(
            per_feat_values=self.linear_correlations,
            out_file=out_file
        )
        print("You can also access these results via the class attribute: 'linear_correlations'.")
