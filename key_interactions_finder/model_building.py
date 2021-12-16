"""
Module to prepare and run machine learning in either a supervised or unsupervised fashion.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import json
import pickle
import pandas as pd
import numpy as np


# sklearn learn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.decomposition import PCA

# sklearn bits and bobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# own module.
from key_interactions_finder.utils import _prep_out_dir


@dataclass
class MachineLearnModel(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised ML models."""

    @abstractmethod
    def build_models(self, save_models=True):
        """Runs the machine learning and summarizes the results."""

    @abstractmethod
    def _describe_ml_planned(self):
        """Prints to the user a summary of what machine learning they are about to perform."""

    @abstractmethod
    def _assign_model_params(self):
        """Assigns the grid search paramters for the ML based on user criteria."""

    def _save_best_models(self, best_model, out_path):
        """Save the best performing model to disk."""
        with open(out_path, 'wb') as file_out:
            pickle.dump(best_model, file_out)
        return print(f"Model saved to disk at: {out_path} ")


@dataclass
class SupervisedModel(MachineLearnModel):
    """Class to Construct Supervised Machine Learning Models."""

    dataset: pd.DataFrame
    evaluation_split_ratio: float = 0.15
    classes_to_use: list = field(default_factory=[])
    scaling_method: str = "min_max"
    out_dir: str = ""
    cross_validation_splits: int = 5
    cross_validation_repeats: int = 3
    search_approach: str = "quick"  # none, quick, moderate or exhaustive

    # Dynamically generated:
    model_params: dict = field(init=False)
    cv: RepeatedStratifiedKFold = field(init=False)
    feat_names: np.ndarray = field(init=False)
    train_data_scaled: np.ndarray = field(init=False)
    eval_data_scaled: np.ndarray = field(init=False)
    y_train: pd.core.series.Series = field(init=False)
    y_eval: pd.core.series.Series = field(init=False)
    ml_models: dict = field(init=False)

    if scaling_method not in ["min_max", "standard_scaling"]:
        raise AssertionError(
            "Please set the scaling_method to be either 'min_max' or 'standard_scaling'.")

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Setup the provided dataset and params for ML."""
        self.out_dir = _prep_out_dir(self.out_dir)

        # Filter to only include desired classes.
        if len(self.classes_to_use) != 0:
            self.dataset = self.dataset[self.dataset["Classes"].isin(
                self.classes_to_use)]

        # Train-test split
        df_features = self.dataset.drop("Classes", axis=1)
        x_array = df_features.to_numpy()
        self.feat_names = df_features.columns.values
        y = self.dataset["Classes"]
        x_array_train, x_array_eval, self.y_train, self.y_eval = train_test_split(
            x_array, y, test_size=self.evaluation_split_ratio)

        self.train_data_scaled, self.eval_data_scaled = self._supervised_scale_features(
            x_array_train=x_array_train, x_array_eval=x_array_eval)

        # Define ML Pipeline:
        self.cv = RepeatedStratifiedKFold(
            n_splits=self.cross_validation_splits, n_repeats=self.cross_validation_repeats)
        self.model_params = self._assign_model_params()

        print(self._describe_ml_planned())

    def _supervised_scale_features(self, x_array_train, x_array_eval):
        """Scale all features with either MinMaxScaler or StandardScaler Scaler.
        implementation for supervised and unsupervised is different to prevent
        information leakage when doing supervised learning."""
        if self.scaling_method == "min_max":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(x_array_train)
        train_data_scaled = scaler.transform(x_array_train)
        eval_data_scaled = scaler.transform(x_array_eval)

        return train_data_scaled, eval_data_scaled

    def build_models(self, save_models=True):
        """Runs the machine learning and summarizes the results."""
        scores = []
        self.ml_models = {}

        if save_models:
            if not os.path.exists("temporary_files"):
                os.makedirs("temporary_files")
            np.save("temporary_files/feature_names.npy", self.feat_names)
            np.save("temporary_files/y_train.npy", self.y_train)
            np.save("temporary_files/y_eval.npy", self.y_eval)
            np.save("temporary_files/train_data_scaled.npy",
                    self.train_data_scaled)
            np.save("temporary_files/eval_data_scaled.npy",
                    self.eval_data_scaled)

        for model_name, mp in self.model_params.items():
            clf = GridSearchCV(mp['model'], mp['params'],
                               cv=self.cv, refit=True)
            clf.fit(self.train_data_scaled, self.y_train)
            scores.append({
                'model': model_name,
                'best_params': clf.best_params_,
                'best_score': clf.best_score_,
                'best_std': clf.cv_results_['std_test_score'][clf.best_index_]
            })
            self.ml_models[model_name] = clf

            if save_models:
                temp_out_path = "temporary_files" + "/" +  \
                    str(model_name) + "_Model.pickle"
                self._save_best_models(
                    best_model=clf.best_estimator_, out_path=temp_out_path)

        # Provide a model summary with the train/test data.
        print("Model building complete, final results below:")
        print("")
        print(pd.DataFrame(scores, columns=[
            'model', 'best_params', 'best_score', 'best_std']))

    def evaluate_model(self):
        """Evaluates model performance on the validation data set."""
        for model_name, clf in self.ml_models.items():
            print(f"Classification report for: {model_name}")
            yhat = clf.predict(self.eval_data_scaled)
            print(classification_report(self.y_eval, yhat))

    def _assign_model_params(self):
        """Assigns the grid search paramters for the ML based on user criteria."""
        model_params = {
            "ada_boost": {"model": AdaBoostClassifier(), "params": {}},
            "random_forest": {"model": RandomForestClassifier(), "params": {}},
            "GBoost": {"model": GradientBoostingClassifier(), "params": {}}
        }

        with open("key_interactions_finder/model_parameters.json") as file_in:
            all_hyper_params = json.load(file_in)

        # Assign search parameters accordingly.
        if self.search_approach == "none":
            model_params["ada_boost"]["params"] = all_hyper_params["none"]["ada_boost"]["params"]
            model_params["random_forest"]["params"] = all_hyper_params["none"]["random_forest"]["params"]
            model_params["GBoost"]["params"] = all_hyper_params["none"]["GBoost"]["params"]
        elif self.search_approach == "quick":
            model_params["ada_boost"]["params"] = all_hyper_params["quick"]["ada_boost"]["params"]
            model_params["random_forest"]["params"] = all_hyper_params["quick"]["random_forest"]["params"]
            model_params["GBoost"]["params"] = all_hyper_params["quick"]["GBoost"]["params"]
        elif self.search_approach == "moderate":
            model_params["ada_boost"]["params"] = all_hyper_params["moderate"]["ada_boost"]["params"]
            model_params["random_forest"]["params"] = all_hyper_params["moderate"]["random_forest"]["params"]
            model_params["GBoost"]["params"] = all_hyper_params["moderate"]["GBoost"]["params"]
        elif self.search_approach == "exhaustive":
            model_params["ada_boost"]["params"] = all_hyper_params["exhaustive"]["ada_boost"]["params"]
            model_params["random_forest"]["params"] = all_hyper_params["exhaustive"]["random_forest"]["params"]
            model_params["GBoost"]["params"] = all_hyper_params["exhaustive"]["GBoost"]["params"]
        else:
            raise ValueError(
                "You must select either 'none' 'quick', 'moderate' or 'exhaustive' for the search_approach option.")

        return model_params

    def _describe_ml_planned(self):
        """Prints out a summary to the user of what machine learning protoctol they have selected."""
        eval_pcent = self.evaluation_split_ratio*100
        train_pcent = 100 - eval_pcent

        out_text = "\n"
        out_text += "Below is a summary of the machine learning you have planned.\n"

        out_text += f"You will use {self.cross_validation_splits}-fold cross validation "
        out_text += f"and perform {self.cross_validation_repeats} repeats.\n"

        out_text += f"{train_pcent}% of your data will be used for training, "
        out_text += f"which is {len(self.train_data_scaled)} observations.\n"

        out_text += f"{eval_pcent}% of your data will be used for evaluating the best models "
        out_text += f"from cross validation, which is {len(self.eval_data_scaled)} observations.\n"

        # TODO.
        out_text += f"You will evaluate the following models: and an exhausitive search patten.\n"

        out_text += "If you're happy with the above, lets get model building!"
        return out_text


# Can have a go with PCA Maybe or maybe just not use as already have a lot of stuff...
@dataclass
class UnsupervisedModel(MachineLearnModel):
    """Class to Construct Unsupervised Machine Learning Models."""
    dataset: pd.DataFrame
    out_dir: str = ""

    # Dynamically generated:
    feat_names: np.ndarray = field(init=False)
    data_scaled: np.ndarray = field(init=False)
    ml_models: dict = field(init=False)

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Setup the provided dataset and params for ML."""
        self.out_dir = _prep_out_dir(self.out_dir)

        # Allows a user with supervised dataset to use this method.
        try:
            self.dataset = self.dataset.drop(["Classes"], axis=1)
        except KeyError:
            pass

        self.feat_names = self.dataset.columns.values
        data_array = self.dataset.to_numpy()

        scaler = StandardScaler()
        self.data_scaled = scaler.fit_transform(data_array)

        print(self._describe_ml_planned())

    def build_models(self, save_models=True):
        """Runs the machine learning and summarizes the results."""
        self.ml_models = {}

        pca = PCA()
        pca.fit(self.data_scaled)
        self.ml_models["PCA"] = pca
        print("All models built.")

    def _describe_ml_planned(self):
        """Prints out a summary to the user of what machine learning protoctol they have selected. """
        out_text = "\n"
        out_text += "Below is a summary of the unsupervised machine learning you have planned. \n"

        out_text += f"All of your data will be used for training the model, "
        out_text += f"which is {len(self.dataset)} observations.\n"

        out_text += f"Currently you will use just PCA to get your results. "
        out_text += f"More methods might be added in the future. "  # TODO.

        out_text += "If you're happy with the above, lets get model building!"
        return out_text

    def _assign_model_params(self):
        """Assigns the grid search paramters for the ML based on user criteria."""
        # Currently not needed, maybe remove as abstract method then? # TODO.
