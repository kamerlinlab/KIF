"""
Module to prepare and run machine learning in either a supervised or unsupervised fashion.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import json
import pickle
from typing import Tuple
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

from key_interactions_finder.utils import _prep_out_dir


@dataclass
class MachineLearnModel(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised ML models."""

    @abstractmethod
    def build_models(self, save_models=True):
        """Runs the machine learning and summarizes the results."""

    @abstractmethod
    def _describe_ml_planned(self):
        """Provide a summary to the user of what machine learning protoctol they have selected."""

    @staticmethod
    def _save_best_models(best_model, out_path: str) -> None:
        """Save the best performing model to disk."""
        with open(out_path, 'wb') as file_out:
            pickle.dump(best_model, file_out)
        print(f"Model saved to disk at: {out_path}")


@dataclass
class SupervisedModel(MachineLearnModel):
    """
    Class to Construct Supervised Machine Learning Models.

    Attributes
    ----------
    dataset : pd.DataFrame
        Input dataset.

    classes_to_use : list
        Names of the classes to train the model on. Must be a list of two strings.

    evaluation_split_ratio : float
        Ratio of data that should be used to make the evaluation test set.
        The rest of the data will be used for the training/hyper-param tuning.
        Default = 0.15

    scaling_method : str
        How to scale the dataset prior to machine learning.
        Options are "min_max" (scikit-learn's MinMaxScaler)
        or "standard_scaling" (scikit-learn's StandardScaler).
        Default = "min_max"

    out_dir : str
        Directory path to store results files to.
        Default = ""

    cross_validation_splits : int
        Number of splits for the cross validation. I.E., the "k" in k-fold
        cross validation.
        Default = 5

    cross_validation_repeats :
        Nummer of repeats of the k-fold cross validation to perform.
        Default = 3

    search_approach : str
        Define how extensive the grid search protocol should be for the models.
        Options are: "none", "quick", "moderate" or "exhaustive".
        Default = "quick"

    model_params : dict
        Nested dictionary of model parameters that can be read directly into
        Scikit-learn's implementation of grid search cv.

    cross_validation_approach : RepeatedStratifiedKFold
        Instance of scikit-learn's RepeatedStratifiedKFold class for model building.

    feat_names : np.ndarray
        All feature names/labels.

    train_data_scaled : np.ndarray
        Training data scaled with either "min_max" or "standard_scaling".

    eval_data_scaled : np.ndarray
        Data used for evaluation scaled with either "min_max" or "standard_scaling".

    y_train : pd.Series
        Class labels for the training/testing data.

    y_eval : pd.Series
        Class labels for the validation data.

    ml_models : dict
        Keys are the model name/method and values are the instance of the
        built model.

    Methods
    -------
    build_models(save_models)
        Runs the machine learning and summarizes the results.

    evaluate_model()
        Evaluates model performance on the validation data set and
        prints a summary of this to the screen.
    """
    dataset: pd.DataFrame
    classes_to_use: list
    evaluation_split_ratio: float = 0.15
    scaling_method: str = "min_max"
    out_dir: str = ""
    cross_validation_splits: int = 5
    cross_validation_repeats: int = 3
    search_approach: str = "quick"  # none, quick, moderate or exhaustive

    model_params: dict = field(init=False)
    cross_validation_approach: RepeatedStratifiedKFold = field(init=False)
    feat_names: np.ndarray = field(init=False)
    train_data_scaled: np.ndarray = field(init=False)
    eval_data_scaled: np.ndarray = field(init=False)
    y_train: pd.Series = field(init=False)
    y_eval: pd.Series = field(init=False)
    ml_models: dict = field(init=False)

    if scaling_method not in ["min_max", "standard_scaling"]:
        raise AssertionError(
            "Please set the scaling_method to be either 'min_max' or 'standard_scaling'.")

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Setup the provided dataset and params for ML."""
        self.out_dir = _prep_out_dir(self.out_dir)
        self.ml_models = {}  # not populated till method called later.

        # Filter to only include desired classes.
        if len(self.classes_to_use) != 0:
            self.dataset = self.dataset[self.dataset["Classes"].isin(
                self.classes_to_use)]

        # Train-test split
        df_features = self.dataset.drop("Classes", axis=1)
        x_array = df_features.to_numpy()
        self.feat_names = df_features.columns.values
        y_classes = self.dataset["Classes"]
        x_array_train, x_array_eval, self.y_train, self.y_eval = train_test_split(
            x_array, y_classes, test_size=self.evaluation_split_ratio)

        self.train_data_scaled, self.eval_data_scaled = self._supervised_scale_features(
            x_array_train=x_array_train, x_array_eval=x_array_eval)

        # Define ML Pipeline:
        self.cross_validation_approach = RepeatedStratifiedKFold(
            n_splits=self.cross_validation_splits, n_repeats=self.cross_validation_repeats)
        self.model_params = self._assign_model_params()

        print(self._describe_ml_planned())

    def _supervised_scale_features(self,
                                   x_array_train: np.ndarray,
                                   x_array_eval: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale all features with either MinMaxScaler or StandardScaler Scaler.
        implementation for supervised and unsupervised is different to prevent
        information leakage when doing supervised learning.

        Parameters
        ----------
        x_array_train : np.ndarray
            Subset of feature data for training/testing with grid search cv.

        x_array_eval : np.ndarray
            Subset of feature data for model validation.

        Returns
        ----------
        train_data_scaled : np.ndarray
            Scaled training/testing data ready for model building.

        eval_data_scaled : np.ndarray
            Scaled training/testing data ready for model validation.
        """
        if self.scaling_method == "min_max":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(x_array_train)
        train_data_scaled = scaler.transform(x_array_train)
        eval_data_scaled = scaler.transform(x_array_eval)

        return train_data_scaled, eval_data_scaled

    def build_models(self, save_models: bool = True) -> None:
        """
        Runs the machine learning and summarizes the results.

        Parameters
        ----------
        save_models : bool
            Whether to save the ML models made to disk.
        """
        scores = []
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

        for model_name, mod_params in self.model_params.items():
            clf = GridSearchCV(mod_params['model'], mod_params['params'],
                               cv=self.cross_validation_approach, refit=True)
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
        print("Model building complete, final results with train/test data below:")
        print("")
        print(pd.DataFrame(scores, columns=[
            'model', 'best_params', 'best_score', 'best_std']))

    def evaluate_model(self) -> None:
        """Evaluates model performance on the validation data set and
        prints a summary of this to the screen."""
        for model_name, clf in self.ml_models.items():
            print(f"Classification report for: {model_name}")
            yhat = clf.predict(self.eval_data_scaled)
            print(classification_report(self.y_eval, yhat))

    def _assign_model_params(self) -> dict:
        """
        Assigns the grid search paramters for the ML based on user criteria.

        Returns
        ----------
        dict
            Nested dictionary of model parameters that can be read directly into
            Scikit-learn's implementation of grid search cv.
        """
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
            error_message = ("You must select either 'none' 'quick', 'moderate' "
                             "or 'exhaustive' for the search_approach option.")
            raise ValueError(error_message)

        return model_params

    # def _assign_search_params(self, ml_algo: str) -> dict:
    #     """

    #     Returns
    #     ----------
    #     dict
    #         Dictionary of hyperparameters to tune for a specific ML model.
    #     """
    #     hyper_params_file = "key_interactions_finder/model_params/gridsearch_" + \
    #         str(self.search_approach) + ".json"
    #     with open(hyper_params_file) as file_in:
    #         hyper_params = json.load(file_in)

    def _describe_ml_planned(self) -> str:
        """
        Provide a summary to the user of what machine learning protoctol they have selected.

        Returns
        ----------
        str
            Summary text.
        """
        eval_pcent = self.evaluation_split_ratio*100
        train_pcent = 100 - eval_pcent

        out_text = "\n"
        out_text += "Below is a summary of the machine learning you have planned.\n"

        out_text += f"You will use {self.cross_validation_splits}-fold cross validation "
        out_text += f"and perform {self.cross_validation_repeats} repeats.\n"

        out_text += f"You will use {len(self.dataset.columns)} features to build the model, with "
        out_text += f"{train_pcent}% of your data used for training the model, "
        out_text += f"which is {len(self.train_data_scaled)} observations. \n"

        out_text += f"{eval_pcent}% of your data will be used for evaluating the best models produced "
        out_text += f"by cross validation, which is {len(self.eval_data_scaled)} observations.\n"

        out_text += "If you're happy with the above, lets get model building!"
        return out_text


# Can have a go with PCA Maybe or maybe just not use as already have a lot of stuff...
@dataclass
class UnsupervisedModel(MachineLearnModel):
    """
    Class to Construct Unsupervised Machine Learning Models.

    At present, there is limited support for this module with only
    principal component analysis (PCA) available.

    Attributes
    ----------
    dataset : pd.DataFrame
        Input dataset.

    out_dir : str
        Directory path to store results files to.
        Default = ""

    feat_names : np.ndarray
        All feature names/labels.

    data_scaled : np.ndarray
        Input dataset scaled with Standard scaling.

    ml_models : dict
        Keys are the model name/method and values are the instance of the
        built model.

    Methods
    -------
    build_models(save_models)
        Runs the machine learning and summarizes the results.
    """
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

        self.ml_models = {}  # not populated till method called later.

        # Allows a user with a supervised dataset to use this method.
        try:
            self.dataset = self.dataset.drop(["Classes"], axis=1)
        except KeyError:
            pass

        self.feat_names = self.dataset.columns.values
        data_array = self.dataset.to_numpy()

        scaler = StandardScaler()
        self.data_scaled = scaler.fit_transform(data_array)

        print(self._describe_ml_planned())

    # TODO - SAVE MODELS.
    def build_models(self, save_models: bool = True) -> None:
        """
        Runs the machine learning and summarizes the results.

        Parameters
        ----------
        save_models : bool
            Whether to save the ML models made to disk.
            Default is True.
        """
        pca = PCA()
        pca.fit(self.data_scaled)
        self.ml_models["PCA"] = pca
        print("All models built.")

    def _describe_ml_planned(self) -> str:
        """
        Provide a summary to the user of what machine learning protoctol they have selected.

        Returns
        ----------
        str
            Summary text.
        """
        out_text = "\n"
        out_text += "Below is a summary of the unsupervised machine learning you have planned. \n"

        out_text += f"You will use {len(self.dataset.columns)} features to build the model, with "
        out_text += f"all of your data will be used for training the model, "
        out_text += f"which is {len(self.dataset)} observations.\n"

        out_text += f"Currently you will use principal component analysis to get your results. "
        out_text += f"More methods might be added in the future. "

        out_text += "If you're happy with the above, lets get model building!"
        return out_text
