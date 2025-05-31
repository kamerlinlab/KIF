"""
Module to prepare and run machine learning (ML) in either a supervised or unsupervised fashion.

3 Classes for end user usage:

1. ClassificationModel
    For building ML models with categorical target data (classification).

2. RegressionModel
    For building ML models with continious target data (regression).

3. UnsupervisedModel
    For building ML models for datasets without labels.
    Limited support for this module at present.


These classes inherit first from an abstract base class called "_MachineLearnModel".
which sets a basic outline for all 3 classes above.

Classes 1 and 2 then inherit from a parent class called "_SupervisedRunner" which abstracts
as much as their shared behaviour as possible.
"""

import contextlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# sklearn bits and bobs
import sklearn.metrics as metrics
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ML models
from xgboost import XGBClassifier, XGBRegressor

from key_interactions_finder import data_preperation
from key_interactions_finder.utils import _prep_out_dir


@dataclass
class _MachineLearnModel(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised ML models."""

    @abstractmethod
    def build_models(self, save_models=True):
        """Runs the machine learning and summarizes the results."""

    @abstractmethod
    def describe_ml_planned(self):
        """Prints a summary of what machine learning protocol has been selected."""

    @staticmethod
    def _save_best_models(best_model, out_path: str) -> None:
        """Saves the best performing model to disk."""
        with open(out_path, "wb") as file_out:
            pickle.dump(best_model, file_out)
        print(f"Model saved to disk at: {out_path}")


@dataclass
class _SupervisedRunner(_MachineLearnModel):
    """
    Abstraction of as much as the shared behaviour for the supervised
    classification and regression classes (the classes the user actually users)
    to prevent duplication.

    Note that there is no __post_init__ method called here because each
    inheriting class is given their own instead.

    Attributes
    ----------

    dataset : pd.DataFrame
        Input dataset.

    models_to_use : list
        List of machine learning models/algorithims to use.
        Default = ["CatBoost"]

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

    cross_validation_splits : Optional[int]
        Number of splits in the cross validation, (the "k" in k-fold cross validation).
        Default = 5

    cross_validation_repeats : Optional[int]
        Number of repeats for the k-fold cross validation to perform.
        Default = 3

    search_approach : Optional[str]
        Define how extensive the grid search protocol should be for the models.
        Options are: "none", "quick", "moderate", "exhaustive" or custom.
        Default = "quick"

    Methods
    -------

    describe_ml_planned()
        Prints a summary of what machine learning protocol has been selected.

    build_models(save_models)
        Runs the machine learning and summarizes the results.

    _assign_model_params(models_to_use, available_models, search_approach, is_classification)
        Assigns the grid search paramters for the ML based on user criteria.

    _supervised_scale_features(scaling_method, x_array_train, x_array_eval)
        Scale all features with either MinMaxScaler or StandardScaler Scaler.

    """

    dataset: pd.DataFrame
    models_to_use: list = field(default_factory=["CatBoost"])
    evaluation_split_ratio: float = 0.15
    scaling_method: str = "min_max"
    out_dir: str = ""
    cross_validation_splits: Optional[int] = 5
    cross_validation_repeats: Optional[int] = 3
    search_approach: Optional[str] = "none"

    # Generated later by method calls.
    available_models: dict = field(init=False)
    ml_models: dict = field(init=False)
    all_model_params: dict = field(init=False)
    ml_datasets: dict = field(init=False)
    cross_validation_approach: RepeatedStratifiedKFold = field(init=False)
    feat_names: np.ndarray = field(init=False)

    if scaling_method not in ["min_max", "standard_scaling"]:
        raise AssertionError("Please set the 'scaling_method' to be either 'min_max' or 'standard_scaling'.")

    # No __post_init__ method because both inheriting classes have their own.

    def describe_ml_planned(self):
        """Prints a summary of what machine learning protocol has been selected."""
        eval_pcent = self.evaluation_split_ratio * 100
        train_pcent = 100 - eval_pcent
        train_obs = len(self.ml_datasets["train_data_scaled"])
        eval_obs = len(self.ml_datasets["eval_data_scaled"])

        out_text = "\n"
        out_text += "Below is a summary of the machine learning you have planned.\n"

        out_text += f"You will use {self.cross_validation_splits}-fold cross validation "
        out_text += f"and perform {self.cross_validation_repeats} repeats.\n"

        out_text += f"You will use up to {len(self.dataset.columns)} features to build each "
        out_text += f"model, with {train_pcent}% of your data used for training the model, "
        out_text += f"which is {train_obs} observations. \n"

        out_text += f"{eval_pcent}% of your data will be used for evaluating the best models "
        out_text += f"produced by the {self.cross_validation_splits}-fold cross validation, "
        out_text += f"which is {eval_obs} observations.\n"

        out_text += f"You have selected to build {len(self.models_to_use)} machine "
        out_text += "learning model(s), with the following hyperparameters: \n \n"

        for model_name, model_params in self.all_model_params.items():
            out_text += f"A {model_name} model, with grid search parameters: \n"
            single_model_params = model_params["params"]
            out_text += f"{single_model_params} \n"
            out_text += "\n"

        out_text += "If you're happy with the above, lets get model building!"
        print(out_text)

    def build_models(self, save_models: bool = True) -> pd.DataFrame:
        """
        Runs the machine learning and summarizes the results.

        Parameters
        ----------

        save_models : bool
            Whether to save the ML models made to disk.

        Returns
        ----------

        pd.DataFrame
            A dataframe with the best score and standard deviation
            (obtained from grid search cv) for each model used.
        """
        scores = []
        for model_name, mod_params in self.all_model_params.items():
            start_time = time.monotonic()

            clf = GridSearchCV(mod_params["model"], mod_params["params"], cv=self.cross_validation_approach, refit=True)
            clf.fit(self.ml_datasets["train_data_scaled"], self.ml_datasets["y_train"])

            end_time = time.monotonic()
            time_taken = timedelta(seconds=end_time - start_time)
            time_taken_mins = round((time_taken.seconds / 60), 2)

            scores.append(
                {
                    "model": model_name,
                    "best_params": clf.best_params_,
                    "best_score": clf.best_score_,
                    "best_standard_deviation": clf.cv_results_["std_test_score"][clf.best_index_],
                    "Time taken to build model (minutes)": time_taken_mins,
                }
            )
            self.ml_models[model_name] = clf

            if save_models:
                temp_folder = Path("temporary_files")
                if not temp_folder.exists():
                    Path.mkdir(temp_folder)

                feat_names_file = Path(temp_folder, "feature_names.npy")
                np.save(feat_names_file, self.feat_names)

                model_file_name = str(model_name) + "_Model.pickle"
                model_out_path = Path(temp_folder, model_file_name)

                self._save_best_models(best_model=clf.best_estimator_, out_path=model_out_path)

        # Provide a model summary with the train/test data.
        print("Model building complete, returning final results with train/test datasets to you.")

        final_results_df = pd.DataFrame(
            scores,
            columns=[
                "model",
                "best_params",
                "best_score",
                "best_standard_deviation",
                "Time taken to build model (minutes)",
            ],
        )

        return final_results_df

    def _assign_model_params(
        self,
        models_to_use: list,
        available_models: dict,
        search_approach: str,
        is_classification: bool,
    ) -> dict:
        """
        Assigns the grid search paramters for the ML based on user criteria.

        Parameters
        ----------

        models_to_use : list[str]
            List of machine learning models/algorithims to use.

        available_models : dict
            Specific ML models available to be used. Keys are the models name and values
            are an instance of the model.

        search_approach : str
            Define how extensive the grid search protocol should be for the models.
            Options are: "none", "quick", "moderate", "exhaustive" or custom.

        is_classification : bool
            If true, the ML model is for classification.
            If false, the ML model is for regression.

        Returns
        ----------

        dict
            Nested dictionary of model parameters that can be read directly into
            Scikit-learn's implementation of grid search cv.
        """
        all_model_params = {}
        for model_name in models_to_use:
            all_model_params[model_name] = available_models[model_name]
            all_model_params[model_name]["params"] = self._assign_search_params(
                is_classification=is_classification, search_approach=search_approach, ml_algo=model_name
            )

        return all_model_params

    @staticmethod
    def _assign_search_params(is_classification: bool, search_approach: str, ml_algo: str) -> dict:
        """
        Define grid search CV params for a specific supervised machine learning algorithim.

        Parameters
        ----------

        is_classification : bool
            If true, the ML model is for classification.
            If false, the ML model is for regression.

        search_approach : str
            Define how extensive the grid search protocol should be for the models.
            Options are: "none", "quick", "moderate", "exhaustive" or "custom".

        ml_algo : str
            Name of the machine learning algorithim.

        Returns
        ----------

        dict
            Dictionary of hyperparameters to tune for a specific ML algorithim.
        """
        # doing this so the .json files can be found regardless of users working directory
        # location. There might be a better way to do this...

        file_name = "gridsearch_" + search_approach + ".json"
        module_path = Path(data_preperation.__file__).parent
        hyper_params_file = Path(module_path, "model_params", file_name)

        with open(hyper_params_file, encoding="utf-8") as file_in:
            hyper_params = json.load(file_in)

        if is_classification:
            return hyper_params["classification_models"][ml_algo]["params"]

        return hyper_params["regression_models"][ml_algo]["params"]

    @staticmethod
    def _supervised_scale_features(
        scaling_method: str, x_array_train: np.ndarray, x_array_eval: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale all features with either MinMaxScaler or StandardScaler Scaler.
        implementation for supervised and unsupervised is different to prevent
        information leakage when doing supervised learning.

        Parameters
        ----------

        scaling_method : str
            How to scale the dataset prior to machine learning.
            Options are "min_max" (scikit-learn's MinMaxScaler)
            or "standard_scaling" (scikit-learn's StandardScaler).

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
        if scaling_method == "min_max":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(x_array_train)
        train_data_scaled = scaler.transform(x_array_train)
        eval_data_scaled = scaler.transform(x_array_eval)

        return train_data_scaled, eval_data_scaled


@dataclass
class ClassificationModel(_SupervisedRunner):
    """
    Class to construct supervised machine learning models when the target class
    is categorical (aka classification).

    Attributes
    ----------

    dataset : pd.DataFrame
        Input dataset.

    classes_to_use : list
        Names of the classes to train the model on.
        Can be left empty if you want to use all the classes you currently have.
        Default = [] (use all classes in the Target column.)

    models_to_use : list
        List of machine learning models/algorithims to use.
        Default = ["CatBoost"]

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
        Number of splits in the cross validation, (the "k" in k-fold cross validation).
        Default = 5

    cross_validation_repeats : int
        Number of repeats for the k-fold cross validation to perform.
        Default = 3

    search_approach : str
        Define how extensive the grid search protocol should be for the models.
        Options are: "none", "quick", "moderate", "exhaustive" or "custom".
        Default = "quick"

    use_class_weights : bool
        Choose to weight each class according to the number of observations.
        (This can be used in the case of an imbalanced dataset.)
        The weights that will be used are the inverse of the class distribution so
        each class has effective weight 1 at the end.
        Default = False

    all_model_params : dict
        Nested dictionary of model parameters that can be read directly into
        Scikit-learn's implementation of grid search cv.

    cross_validation_approach : RepeatedStratifiedKFold
        Instance of scikit-learn's RepeatedStratifiedKFold class for model building.

    feat_names : np.ndarray
        All feature names/labels.

    ml_datasets : dict
        Nested dictionary containing the training and testing data (both features and
        classes) needed to run the model building.

    label_encoder : LabelEncoder
        Instance of sci-kit learn's label encoder to encode the target classes.
        Required for the XGBoost model.

    ml_models : dict
        Keys are the model name/method and values are the instance of the
        built model.

    Methods
    -------

    describe_ml_planned()
        Prints a summary of what machine learning protocol has been selected.

    build_models(save_models)
        Runs the machine learning and summarizes the results.

    evaluate_models()
        Evaluates each ML model's performance on the validation data set
        and provides the user with a summary of the results.

    generate_confusion_matrix()
        For each ml model used, determine the confusion matrix from the validation dataset.
    """

    # Only non-shared parameters between classificaiton and regression.
    label_encoder: LabelEncoder = LabelEncoder()
    classes_to_use: list = field(default_factory=[])
    use_class_weights: bool = False

    def __post_init__(self):
        """Setup the provided dataset and params for ML."""
        self.out_dir = _prep_out_dir(self.out_dir)

        # not populated till build_models method called later.
        self.all_model_params = {}
        self.ml_models = {}

        # Filter to only include desired classes.
        if len(self.classes_to_use) != 0:
            self.dataset = self.dataset[self.dataset["Target"].isin(self.classes_to_use)]

        # Train-test splitting and scaling.
        df_features = self.dataset.drop("Target", axis=1)
        x_array = df_features.to_numpy()
        self.feat_names = df_features.columns.values

        # label encode the target - for XGBOOST compatability.
        self.label_encoder = LabelEncoder()
        y_classes = self.label_encoder.fit_transform(self.dataset.Target.values)

        x_array_train, x_array_eval, y_train, y_eval = train_test_split(
            x_array, y_classes, test_size=self.evaluation_split_ratio
        )

        train_data_scaled, eval_data_scaled = self._supervised_scale_features(
            scaling_method=self.scaling_method, x_array_train=x_array_train, x_array_eval=x_array_eval
        )

        self.ml_datasets = {}
        self.ml_datasets["train_data_scaled"] = train_data_scaled
        self.ml_datasets["eval_data_scaled"] = eval_data_scaled
        self.ml_datasets["y_train"] = y_train
        self.ml_datasets["y_eval"] = y_eval

        if self.use_class_weights:
            # For CatBoostClassifier
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights, strict=True))

            # For XGBClassifier, I need the total number of examples in the majority class
            # divided by the total number of examples in the minority class.
            majority = self.dataset["Target"].value_counts()[0]
            minority = self.dataset["Target"].value_counts()[1]
            scaled_weight = round((majority / minority), 2)

            self.available_models = {
                "CatBoost": {
                    "model": CatBoostClassifier(class_weights=class_weights, logging_level="Silent"),
                    "params": {},
                },
                "XGBoost": {
                    "model": XGBClassifier(eval_metric="logloss", scale_pos_weight=scaled_weight),
                    "params": {},
                },
                "Random_Forest": {"model": RandomForestClassifier(class_weight="balanced"), "params": {}},
            }

            print("Class weights have now been added to your dataset.")
            print(f"The class imbalance in your dataset was: 1 : {1 / scaled_weight:.2f}")

        else:
            self.available_models = {
                "CatBoost": {"model": CatBoostClassifier(logging_level="Silent"), "params": {}},
                "XGBoost": {"model": XGBClassifier(eval_metric="logloss"), "params": {}},
                "Random_Forest": {"model": RandomForestClassifier(), "params": {}},
            }

        # Define ML Pipeline:
        self.cross_validation_approach = RepeatedStratifiedKFold(
            n_splits=self.cross_validation_splits, n_repeats=self.cross_validation_repeats
        )

        self.all_model_params = self._assign_model_params(
            models_to_use=self.models_to_use,
            available_models=self.available_models,
            search_approach=self.search_approach,
            is_classification=True,
        )

        self.describe_ml_planned()

    def evaluate_models(self) -> dict:
        """
        Evaluates each ML model's performance on the validation data set
        and provides the user with a summary of the results.

        Returns
        ----------

        dict
            A dictionary with keys being the model names and values being a pd.DataFrame
            with several scoring metrics output for each model used.
        """
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        # yhat_decoded = self.label_encoder.inverse_transform(yhat)
        y_eval_decoded = self.label_encoder.inverse_transform(self.ml_datasets["y_eval"])
        class_labels = (np.unique(y_eval_decoded)).tolist()

        all_classification_reports = {}
        for model_name, clf in self.ml_models.items():
            yhat = clf.predict(self.ml_datasets["eval_data_scaled"])
            yhat_decoded = self.label_encoder.inverse_transform(yhat)

            report = metrics.classification_report(y_eval_decoded, yhat_decoded, output_dict=True)

            # now reformat report so dataframe friendly.
            new_report = {}
            for label in class_labels:
                new_report.update({label: report[label]})

            accuracy_row = {
                "precision": "N/A",
                "recall": "N/A",
                "f1-score": report["accuracy"],
                "support": report["weighted avg"]["support"],
            }
            new_report.update({"accuracy": accuracy_row})

            new_report.update({"macro avg": report["macro avg"]})
            new_report.update({"weighted avg": report["weighted avg"]})

            df_classification_report = pd.DataFrame(new_report).transpose()

            all_classification_reports.update({model_name: df_classification_report})

        print("Returning classification reports for each model inside a single dictionary")
        return all_classification_reports

    def generate_confusion_matrix(self) -> dict:
        """
        For each ml model used, determine the confusion matrix from the validation data set.
        Returns a dictionary with model names as keys and the corresponding matrix as the values.

        Returns
        ----------

        dict
            Keys are strings of each model name. Values are the confusion matrix
            of said model as a numpy.ndarray.
        """
        confusion_matrices = {}
        for model_name, clf in self.ml_models.items():
            yhat = clf.predict(self.ml_datasets["eval_data_scaled"])
            y_true = self.ml_datasets["y_eval"]

            yhat_decoded = self.label_encoder.inverse_transform(yhat)
            y_true_decoded = self.label_encoder.inverse_transform(y_true)

            confuse_matrix = metrics.confusion_matrix(y_true_decoded, yhat_decoded)

            confusion_matrices.update({model_name: confuse_matrix})

        return confusion_matrices


@dataclass
class RegressionModel(_SupervisedRunner):
    """
    Class to construct supervised machine learning models when the target class
    is contionous (aka regression).

    Attributes
    ----------

    dataset : pd.DataFrame
        Input dataset.

    models_to_use : list
        List of machine learning models/algorithims to use.
        Default = ["CatBoost"]

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
        Number of splits in the cross validation, (the "k" in k-fold cross validation).
        Default = 5

    cross_validation_repeats : int
        Number of repeats for the k-fold cross validation to perform.
        Default = 3

    search_approach : str
        Define how extensive the grid search protocol should be for the models.
        Options are: "none", "quick", "moderate", "exhaustive" or "custom".
        Default = "quick"

    all_model_params : dict
        Nested dictionary of model parameters that can be read directly into
        Scikit-learn's implementation of grid search cv.

    cross_validation_approach : RepeatedStratifiedKFold
        Instance of scikit-learn's RepeatedStratifiedKFold class for model building.

    feat_names : np.ndarray
        All feature names/labels.

    ml_datasets : dict
        Nested dictionary containing the training and testing data (both features and
        classes) needed to run the model building.

    ml_models : dict
        Keys are the model name/method and values are the instance of the
        built model.

    Methods
    -------

    describe_ml_planned()
        Prints a summary of what machine learning protocol has been selected.

    build_models(save_models)
        Runs the machine learning and summarizes the results.

    evaluate_models()
        Evaluates each ML model's performance on the validation data set
        and provides the user with a summary of the results.
    """

    available_models = {
        "CatBoost": {"model": CatBoostRegressor(logging_level="Silent"), "params": {}},
        "XGBoost": {"model": XGBRegressor(objective="reg:squarederror"), "params": {}},
        "Random_Forest": {"model": RandomForestRegressor(), "params": {}},
    }

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Setup the provided dataset and params for ML."""
        self.out_dir = _prep_out_dir(self.out_dir)

        # These are all not populated till a method is called later.
        self.ml_models = {}
        self.all_model_params = {}

        # Train-test splitting and scaling.
        df_features = self.dataset.drop("Target", axis=1)
        x_array = df_features.to_numpy()
        self.feat_names = df_features.columns.values
        y_classes = self.dataset["Target"]
        x_array_train, x_array_eval, y_train, y_eval = train_test_split(
            x_array, y_classes, test_size=self.evaluation_split_ratio
        )

        train_data_scaled, eval_data_scaled = self._supervised_scale_features(
            scaling_method=self.scaling_method, x_array_train=x_array_train, x_array_eval=x_array_eval
        )

        self.ml_datasets = {}
        self.ml_datasets["train_data_scaled"] = train_data_scaled
        self.ml_datasets["eval_data_scaled"] = eval_data_scaled
        self.ml_datasets["y_train"] = y_train
        self.ml_datasets["y_eval"] = y_eval

        # Define ML Pipeline:
        self.cross_validation_approach = RepeatedKFold(
            n_splits=self.cross_validation_splits, n_repeats=self.cross_validation_repeats
        )

        self.all_model_params = self._assign_model_params(
            models_to_use=self.models_to_use,
            available_models=self.available_models,
            search_approach=self.search_approach,
            is_classification=False,
        )

        self.describe_ml_planned()

    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluates each ML model's performance on the validation data set
        and provides the user with a summary of the results.

        Returns
        ----------

        pd.DataFrame
            Dataframe with each row a containing several regression metrics
            for each ML model generated.
        """
        all_regression_dfs = []
        for model_name, clf in self.ml_models.items():
            y_validation = self.ml_datasets["y_eval"]
            yhat = clf.predict(self.ml_datasets["eval_data_scaled"])

            regression_df = self._regression_metrics(model_name=model_name, y_true=y_validation, y_pred=yhat)

            all_regression_dfs.append(regression_df)

        return pd.concat(all_regression_dfs).reset_index(drop=True)

    @staticmethod
    def _regression_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Calculate several regression statistics for a given ml model.

        Adapted from:
        https://stackoverflow.com/questions/26319259/how-to-get-a-regression-summary-in-scikit-learn-like-r-does

        Parameters
        ----------

        model_name : str
            Name of the ML model.

        y_true : np.ndarray
            1D array of the actual values of the target label.

        y_pred: np.ndarray
            1D array of the predicted values of the target label.

        Returns
        ----------

        pd.DataFrame
            Contains various regression metrics for the provided model.
        """
        explained_variance = np.round(metrics.explained_variance_score(y_true, y_pred), 4)

        mean_absolute_error = np.round(metrics.mean_absolute_error(y_true, y_pred), 4)

        mse = np.round(metrics.mean_squared_error(y_true, y_pred), 4)

        rmse = np.round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 4)

        r_squared = np.round(metrics.r2_score(y_true, y_pred), 4)

        try:
            mean_squared_log_error = np.round(metrics.mean_squared_log_error(y_true, y_pred), 4)

        except ValueError:
            print("""Mean Squared Log Error cannot be calculated as your target column contains
                  negative numbers. Continuing with the other metrics.""")
            mean_squared_log_error = "N/A"

        all_metrics = [
            [model_name, explained_variance, mean_absolute_error, mse, rmse, mean_squared_log_error, r_squared]
        ]

        column_labels = [
            "Model",
            "Explained Variance",
            "Mean Absolute Error",
            "MSE",
            "RMSE",
            "Mean Squared Log Error",
            "r squared",
        ]

        return pd.DataFrame(all_metrics, columns=column_labels)


@dataclass
class UnsupervisedModel(_MachineLearnModel):
    """
    Class to construct machine learning models for when there is
    no target class available (aka, unsupervised learning).

    At present there is limited support for this, with only
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

    describe_ml_planned()
        Prints a summary of what machine learning protocol has been selected.

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

        self.ml_models = {}

        # Allow a user with a supervised dataset to do unsupervised learning.
        with contextlib.suppress(KeyError):
            self.dataset = self.dataset.drop(["Target"], axis=1)

        self.feat_names = self.dataset.columns.values
        data_array = self.dataset.to_numpy()

        scaler = StandardScaler()
        self.data_scaled = scaler.fit_transform(data_array)

        self.describe_ml_planned()

    def describe_ml_planned(self) -> None:
        """Prints a summary of what machine learning protocol has been selected."""
        out_text = "\n"
        out_text += "Below is a summary of the unsupervised machine learning you have planned. \n"

        out_text += f"You will use {len(self.dataset.columns)} features to build the model, with "
        out_text += "all of your data will be used for training the model, "
        out_text += f"which is {len(self.dataset)} observations.\n"

        out_text += "Currently you will use principal component analysis to get your results. "
        out_text += "More methods might be added in the future. "

        out_text += "If you're happy with the above, lets get model building!"
        return out_text

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

        if save_models:
            temp_folder = Path("temporary_files")
            if not temp_folder.exists():
                Path.mkdir(temp_folder)

            feat_names_file = Path(temp_folder, "feature_names.npy")
            np.save(feat_names_file, self.feat_names)

            model_out_path = Path(temp_folder, "PCA_Model.pickle")
            self._save_best_models(best_model=pca, out_path=model_out_path)
