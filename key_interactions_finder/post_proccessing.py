"""
Handels feature importance analysis of the ML datasets.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import csv
import pickle
import pandas as pd
import numpy as np


@dataclass
class PostProcessor(ABC):
    """abstract base class to unify the different postprocessing types."""

    # generic params can go here.
    out_dir: str = ""
    load_from_disk: bool = True  # Need to add method for if false... TODO
    feat_names: np.ndarray = field(init=False)
    best_models: dict = field(init=False)

    all_feat_importances: dict = field(init=False)
    sphere_sizes: dict = field(init=False)

    @abstractmethod
    def get_feature_importance(self):
        """Gets feature importances."""

    @abstractmethod
    def per_res_importance(self):
        """Projects feature importances onto the per-residue level"""

    @abstractmethod
    def _load_data_from_disk(self):
        """Loads previously made ML models from disk into the class."""


@dataclass
class SupervisedPostProcessor(PostProcessor):
    """Class Docstring"""
    # Can move generic params to parent class later...
    y_train: np.ndarray = field(init=False)
    y_eval: np.ndarray = field(init=False)
    train_data_scaled: np.ndarray = field(init=False)
    eval_data_scaled: np.ndarray = field(init=False)

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Read in extra params from disk."""
        self.all_feat_importances = {}

        if self.out_dir != "":
            if os.path.exists(self.out_dir) is False:
                os.makedirs(self.out_dir)
            if self.out_dir[-1] != "/":
                self.out_dir += "/"

        if self.load_from_disk:
            try:
                self.best_models = self._load_data_from_disk()
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
                    "I cannot find the files you generated from a prior machine learning run, "
                    "if you have already run the machine learning, make sure you are inside the right "
                    "working directory. you should see a folder named: 'temporary_files' if you are.")

        else:
            print("To do.")
            # Build a load from session option here. - inheritance... TODO

    def _load_data_from_disk(self):
        """Loads previously made ML models from disk into the class."""
        best_models = {}
        for model_name in ["ada_boost", "GBoost", "random_forest"]:  # BAD PRACTICE TODO!!!!
            model = pickle.load(
                open(f"temporary_files/{model_name}_Model.pickle", 'rb'))
            best_models.update({model_name: model})
        return best_models

    def get_feature_importance(self):
        """Gets feature importances and saves them to disk."""
        self.all_feat_importances = {}
        for model_name, model in self.best_models.items():
            raw_importances = list(np.around(model.feature_importances_, 8))
            feat_importances = zip(self.feat_names, raw_importances)
            sort_feat_importances = sorted(
                feat_importances, key=lambda x: x[1], reverse=True)

            self.all_feat_importances.update(
                {model_name: sort_feat_importances})

            output_location = self.out_dir + \
                str(model_name) + "_Feature_Importances.csv"

            with open(output_location, "w", newline="") as out:
                csv_out = csv.writer(out)
                csv_out.writerow(["Feature", "Importance"])
                for row in sort_feat_importances:
                    csv_out.writerow(row)

        return print(f"Feature importances for all models have been saved to disk at the path: '{self.out_dir}'")

    def per_res_importance(self):
        """Projects feature importances onto the per-residue level"""

        # Check if get_feature_importance already run or user skipped this step.
        if len(self.all_feat_importances) == 0:
            self.get_feature_importance()

        self.sphere_sizes = {}
        for model_name, importances in self.all_feat_importances.items():
            df_feat_import = pd.DataFrame(list(importances))
            df_feat_import_res = df_feat_import[0].str.split(" +", expand=True)

            res1, res2, values = [], [], []
            res1 = (df_feat_import_res[0].str.extract("(\d+)")).astype(int)
            res2 = (df_feat_import_res[1].str.extract("(\d+)")).astype(int)
            values = df_feat_import[1]

            per_res_importance = pd.concat(
                [res1, res2, values], axis=1, join="inner")

            per_res_importance.columns = ["Res1", "Res2", "Score"]

            # largest residue number present in features (can stop iterating at this point.)
            max_res = max(per_res_importance[["Res1", "Res2"]].max())
            # Calculate per residue importances.
            res_ids = []
            tot_scores = []
            for i in range(1, max_res+1, 1):
                res_ids.append(i + 1)
                tot_scores.append(
                    per_res_importance.loc[per_res_importance["Res1"] == i, "Score"].sum() +
                    per_res_importance.loc[per_res_importance["Res2"] == i, "Score"].sum(
                    )
                )

            max_ori_score = max(tot_scores)
            # Rescale scores so that new largest has size 1.0 (good for PyMOL sphere representation as well).
            tot_scores_scaled = []
            for i in range(1, max_res, 1):
                tot_scores_scaled.append(tot_scores[i] / max_ori_score)

            spheres = dict(sorted(zip(
                res_ids, tot_scores_scaled), key=lambda x: x[1], reverse=True))

            # nested dict of model sizes.
            self.sphere_sizes.update({model_name: spheres})

            # save each model output to disk.
            out_file = self.out_dir + \
                str(model_name) + "Per_Residue_Importances.csv"
            with open(out_file, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(["Residue Number", 'Normalised Importance'])
                w.writerows(spheres.items())

        print(self.sphere_sizes)  # remove later.
        return print(f"Per residue feature importances for all models have been saved to disk inside the directory: '{self.out_dir}'")


@dataclass
class UnsupervisedPostProcessor(PostProcessor):
    """Class docstring"""

    def get_feature_importance(self):
        """Gets feature importances."""

    def per_res_importance(self):
        """Projects feature importances onto the per-residue level"""

    def _load_data_from_disk(self):
        """Loads previously made ML models from disk into the class."""
