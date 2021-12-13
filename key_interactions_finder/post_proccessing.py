"""
Runs the feature importance analysis from the ML datasets.
"""
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import csv
import pickle
import pandas as pd
import numpy as np
from key_interactions_finder.utils import _prep_out_dir
from key_interactions_finder.model_building import SupervisedModel
from key_interactions_finder.stat_modelling import ProteinStatModel


@dataclass
class PostProcessor(ABC):
    """Abstract base class to unify the different postprocessing types."""

    @abstractmethod
    def get_per_res_importance(self):
        """Projects feature importances onto the per-residue level"""

    def _dict_to_df_feat_importances(self, feat_importances):
        """
        Convert a dictionary of features and feature importances to a dataframe of 3 columns,
        which are: residue 1 number, residue 2 number and importance score for each feature.
        Helper function for determing the per residue importances.

        Parameters
        ----------
        feat_importances : dict
            dictionary containing each feature name (keys) and corresponding importance score (values).

        Returns
        ----------
        pd.core.frame.DataFrame
            dataframe of residue numbers and scores for each feature.

        """
        df_feat_import = pd.DataFrame(feat_importances.items())
        df_feat_import_res = df_feat_import[0].str.split(" +", expand=True)

        res1, res2, values = [], [], []
        res1 = (df_feat_import_res[0].str.extract("(\d+)")).astype(int)
        res2 = (df_feat_import_res[1].str.extract("(\d+)")).astype(int)
        values = df_feat_import[1]

        per_res_import = pd.concat(
            [res1, res2, values], axis=1, join="inner")
        per_res_import.columns = ["Res1", "Res2", "Score"]

        return per_res_import

    def _per_res_importance(self, per_res_import):
        """
        Sums together all the features importances/scores to determine the per-residue value.

        Parameters
        ----------
        per_res_import : pd.core.frame.DataFrame
            Dataframe with columns of both residues numbers and the importance score for
            each feature.

        Returns
        ----------
        dict
            Keys are each residue, values are the residue's relative importance.
        """
        max_res = max(per_res_import[["Res1", "Res2"]].max())
        res_ids = []
        tot_scores = []
        for i in range(1, max_res+1, 1):
            res_ids.append(i + 1)  # weird way to do this, TODO.
            tot_scores.append(
                per_res_import.loc[per_res_import["Res1"] == i, "Score"].sum() +
                per_res_import.loc[per_res_import["Res2"] == i, "Score"].sum())

        # Rescale scores so that new largest has size 1.0 (good for PyMOL sphere representation as well).
        max_ori_score = max(tot_scores)
        tot_scores_scaled = []
        for i in range(1, max_res, 1):
            tot_scores_scaled.append(tot_scores[i] / max_ori_score)

        spheres = dict(sorted(zip(
            res_ids, tot_scores_scaled), key=lambda x: x[1], reverse=True))

        spheres = {keys: np.around(values, 5)
                   for keys, values in spheres.items()}

        return spheres

    def _per_res_importances_to_file(self, spheres: dict, out_file: str) -> None:
        """
        Write out a per residue importances

        Parameters
        ----------
        spheres : dict
            Dictionary of residue numbers and there scores to write to disk.

        out_file : str
            The full path to write the file too.
        """
        with open(out_file, "w", newline="") as file_out:
            csv_out = csv.writer(file_out)
            csv_out.writerow(["Residue Number", "Normalised Importance"])
            csv_out.writerows(spheres.items())
            print(f"{out_file} written to disk.")


@dataclass
class SupervisedPostProcessor(PostProcessor):
    """"Processes supervised machine learning results."""

    supervised_model: Optional[SupervisedModel]
    out_dir: str = ""
    load_from_disk: bool = True
    feat_names: np.ndarray = field(init=False)
    best_models: dict = field(init=False)
    all_feature_importances: dict = field(init=False)
    per_residue_scores: dict = field(init=False)
    y_train: np.ndarray = field(init=False)
    y_eval: np.ndarray = field(init=False)
    train_data_scaled: np.ndarray = field(init=False)
    eval_data_scaled: np.ndarray = field(init=False)

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Read in extra params from either the class or disk."""
        self.all_feature_importances = {}
        self.out_dir = _prep_out_dir(self.out_dir)

        if self.load_from_disk:
            try:
                self.best_models = self._load_models_from_disk()
                self.feat_names = np.load(
                    "temporary_files/feature_names.npy", allow_pickle=True)
                self.y_train = np.load(
                    "temporary_files/y_train.npy", allow_pickle=True)
                self.y_eval = np.load(
                    "temporary_files/y_eval.npy", allow_pickle=True)
                self.train_data_scaled = np.load(
                    "temporary_files/train_data_scaled.npy", allow_pickle=True)
                self.eval_data_scaled = np.load(
                    "temporary_files/eval_data_scaled.npy", allow_pickle=True)
            except:
                raise FileNotFoundError(
                    "I cannot find the files you generated from a prior "
                    "machine learning run, if you have already run the machine learning, "
                    "make sure you are inside the right working directory. You "
                    "should see a folder named: 'temporary_files' if you are.")

        else:
            self.feat_names = self.supervised_model.feat_names
            self.y_train = self.supervised_model.y_train
            self.y_eval = self.supervised_model.y_eval
            self.train_data_scaled = self.supervised_model.train_data_scaled
            self.eval_data_scaled = self.supervised_model.eval_data_scaled
            # need to get the .best_estimator_ attribute to match the "if" path.
            models_made = self.supervised_model.ml_models.keys()
            self.best_models = {}
            for model in models_made:
                self.best_models[model] = (
                    self.supervised_model.ml_models[model].best_estimator_)

    def _load_models_from_disk(self):
        """Loads previously made machine learning models from disk."""
        best_models = {}
        for model_name in ["ada_boost", "GBoost", "random_forest"]:
            model = pickle.load(
                open(f"temporary_files/{model_name}_Model.pickle", 'rb'))
            best_models.update({model_name: model})
        return best_models

    def get_feature_importance(self):
        """Gets feature importances and saves them to disk."""
        self.all_feature_importances = {}
        for model_name, model in self.best_models.items():
            raw_importances = list(np.around(model.feature_importances_, 8))
            feat_importances = zip(self.feat_names, raw_importances)

            sort_feat_importances = dict(sorted(
                feat_importances, key=lambda x: x[1], reverse=True))

            self.all_feature_importances.update(
                {model_name: sort_feat_importances})

            output_location = self.out_dir + \
                str(model_name) + "_Feature_Importances.csv"

            with open(output_location, "w", newline="") as out:
                csv_out = csv.writer(out)
                csv_out.writerow(["Feature", "Importance"])
                for key, value in sort_feat_importances.items():
                    csv_out.writerow([key, np.around(value, 4)])
                print(f"{output_location} written to disk.")
        print("All feature importances written to disk successfully.")

    def get_per_res_importance(self):
        """Projects feature importances onto the per-residue level"""

        # get_feature_importance has to be run before this function.
        if len(self.all_feature_importances) == 0:
            self.get_feature_importance()

        self.per_residue_scores = {}
        for model_name, feat_importances in self.all_feature_importances.items():
            per_res_import = self._dict_to_df_feat_importances(
                feat_importances)
            spheres = self._per_res_importance(per_res_import)

            # Save output to Class and disk.
            self.per_residue_scores.update({model_name: spheres})

            out_file = self.out_dir + \
                str(model_name) + "Per_Residue_Importances.csv"

            self._per_res_importances_to_file(
                spheres=spheres,
                out_file=out_file
            )

        print("All per residue feature importances were written to disk successfully.")


@dataclass
class UnsupervisedPostProcessor(PostProcessor):
    """Processes unsupervised machine learning results."""

    def get_feature_importance(self):
        """Gets feature importances."""

    def get_per_res_importance(self):
        """Projects feature importances onto the per-residue level."""

    def _load_models_from_disk(self):
        """Loads previously made ML models from disk into the class."""


@dataclass
class StatisticalPostProcessor(PostProcessor):
    """"Processes results from the statistical analysis module."""

    stat_model: ProteinStatModel
    out_dir: str = ""

    feature_directions: list = field(init=False)
    per_residue_directions: dict = field(init=False)
    per_residue_js_distances: dict = field(init=False)
    per_residue_mutual_infos: dict = field(init=False)

    # This is called at the end of the dataclass's initialization procedure.

    def __post_init__(self):
        """Read in extra params from either the class or disk."""
        self.out_dir = _prep_out_dir(self.out_dir)
        self.feature_directions = []
        self.per_residue_directions = {}
        self.per_residue_js_distances = {}
        self.per_residue_mutual_infos = {}

    def get_per_res_importance(self, stat_method: str):
        """
        Projects feature importances onto the per-residue level for a single user selected
        statistical method.

        Parameters
        ----------
        stat_method : str
            Define the statistical method that should be used to generate the per
            residue importances.

        Returns
        ----------
        spheres : dict
            Dictionary of each residue and it's relative importance.

        """
        if stat_method == "jenson_shannon":
            model_name = "jenson_shannon"
            per_res_import = self._dict_to_df_feat_importances(
                self.stat_model.js_distances)
            spheres = self._per_res_importance(per_res_import)
            self.per_residue_js_distances = {model_name: spheres}

        elif stat_method == "mutual_information":
            model_name = "mutual_information"
            per_res_import = self._dict_to_df_feat_importances(
                self.stat_model.mutual_infos)
            spheres = self._per_res_importance(per_res_import)
            self.per_residue_mutual_infos = {model_name: spheres}

        else:
            raise ValueError(
                """You did not select one of either 'jenson_shannon'
                 or 'mutual_information' for the 'stat_method' parameter.""")

        out_file = self.out_dir + \
            str(model_name) + "Per_Residue_Importances.csv"

        self._per_res_importances_to_file(
            spheres=spheres,
            out_file=out_file
        )
        return spheres

    def plot_probability_distributions(self):
        """Plots the probablity distributions of the user output."""
        # probablity_distributions

    def estimate_feature_directions(self):
        """
        Estimate the direction each feature favours.

        Incredibly simple logic, so user should be warned about this.
        When they run it.

        I.E., do higher values of this feature tend to be seen in one class or
        vice versa (therefore is it stabilising/destabilising the feature)
        # TODO Add some more description here.
        """
        # self.stat_model.class_names
        avg_contact_scores = {}
        self.feature_directions = []
        for class_name, class_observations in self.stat_model.per_class_datasets.items():
            avg_contact_scores[class_name] = class_observations.mean()
        print(avg_contact_scores)

        class_0_name = self.stat_model.class_names[0]
        class_1_name = self.stat_model.class_names[1]

        for feature_name, class_0_scores in avg_contact_scores[class_0_name].items():
            class_1_score = avg_contact_scores[class_1_name][feature_name]

            if class_0_scores >= class_1_score:
                self.feature_directions.append(class_0_name)
            else:
                self.feature_directions.append(class_1_name)

        # TODO ADD A WARNING HERE ABOUT OUTCOME.
        # TODO SAVE TO FILE.
        print(self.feature_directions)

    def estimate_per_residue_directions(self):
        """
        As above but for per residues, TODO.

        """
