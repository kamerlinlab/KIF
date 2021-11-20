from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import pickle
import os

# sklearn learn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# sklearn bits and bobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from key_interactions_finder.data_preperation import SupervisedFeatureData

# Module prepares for and runs the machine learning in either a supervised or unsupervised fashion. 



###### TODO: 
# 1. Train-test split. 
# 2. scale_features -TRAIN AND TEST BE DONE SEPERATE HERE YEP! 
# https://stackoverflow.com/questions/49444262/normalize-data-before-or-after-split-of-training-and-testing-data 
# 3. Define Models to use.
# Make options to control the intensity of the training. 
# 4. Output 

@dataclass
class MachineLearnModel(ABC):
    """Abstract base class to unify the construction of supervised and unsupervised machine learning models."""
    
    @abstractmethod
    def build_models(self):
        """Call to run the actual model building process."""
        pass


    @abstractmethod
    def _describe_ml_planned(self):
        """Prints to the user a summary of what machine learning they are about to perform."""  
        pass


    def _save_best_models(self, best_model, out_path):
        """Save the best performing model to disk."""
        with open(out_path, 'wb') as f:
            pickle.dump(best_model, f)
        return print(f"Model saved to disk at: {out_path} ")



@dataclass
class SupervisedModel(MachineLearnModel):
    """Class to Construct Supervised Machine Learning Models."""

    # Can move generic params to parent class later... 
    dataset: pd.core.frame.DataFrame
    evaluation_split_ratio: float = 0.15
    scaling_method: str = "min_max"
    out_dir: str = "" 
    cross_validation_splits: int = 5
    cross_validation_repeats: int = 3

    # Fields generated during initialiation below:
    model_params: dict = field(init=False)
    cv: RepeatedStratifiedKFold = field(init=False) # SEEMS A BIT AWKWARD IS THIS THE RIGHT THING TO DO?
    #cv: sklearn.model_selection._split.RepeatedStratifiedKFold = field(init=False)
    train_data_scaled: np.ndarray = field(init=False)
    y_train: pd.core.series.Series = field(init=False)

    eval_data_scaled: np.ndarray = field(init=False)
    y_eval: pd.core.series.Series = field(init=False)

    
    if scaling_method not in ["min_max", "standard_scaling"]:
        raise AssertionError("Please set the scaling_method to be either min_max or standard_scaling")
    


    # This is called at the end of the dataclasses initialization procedure. 
    def __post_init__(self):
        """Setup the provided dataset and params for ML."""

        if self.out_dir != "":
            if os.path.exists(self.out_dir) == False:
                os.makedirs(self.out_dir)
            if self.out_dir[-1] != "/":
                self.out_dir += "/"

        # Train-test split.
        X = self.dataset.drop("Classes", axis=1)
        X_array = X.to_numpy()  
        y = self.dataset["Classes"]
        X_array_train, X_array_eval, self.y_train, self.y_eval = train_test_split(X_array, y, test_size = self.evaluation_split_ratio)

        # Scale features 
        self.train_data_scaled, self.eval_data_scaled = self._supervised_scale_features(
            X_array_train=X_array_train, X_array_eval=X_array_eval)

        # Define ML Pipeline:
        self.cv = RepeatedStratifiedKFold(n_splits=self.cross_validation_splits, n_repeats=self.cross_validation_repeats) 
        # Can read in choice of model_params based on user_specified if - else statement instead? 
        self.model_params = {
            'random_forest': {
                'model': RandomForestClassifier(),
                'params' : {
                    'n_estimators': [2] #[100, 250, 500]
                }
            },
        }

        # Provide overview of what user has planned.
        print(self._describe_ml_planned())
        return None


    def build_models(self, save_models=True):
        """Call to run the actual model building process."""
        scores = []
        ml_models = {}
        # add a dictionary too to store the generated models? 
        print(self.model_params)
        for model_name, mp in self.model_params.items():
            clf =  GridSearchCV(mp['model'], mp['params'], cv=self.cv, refit=True)
            clf.fit(self.train_data_scaled, self.y_train)
            scores.append({
                'model': model_name,
                'best_params': clf.best_params_,
                'best_score': clf.best_score_,
                'best_std': clf.cv_results_['std_test_score'][clf.best_index_]
            })
            ml_models[model_name] = clf
            if save_models == True:
                out_path = self.out_dir + str(model_name) + "_Model.pickle"
                self._save_best_models(best_model=clf.best_estimator_, out_path=out_path)

        # Provide a model summary with the train/test data. 
        print(pd.DataFrame(scores,columns=['model','best_params','best_score','best_std']))
        print(ml_models)
        return None


    def evaluate_model(self):
        """Evaluates model performance on the validation data set."""
        # Evaluate the model on the data reserved for evaluation. 
        # Rename variables to do.
        # yhat = clf_PTP1B_CSP.predict(eval_feat_sets['PTP1B_CSP'])
        # print(classification_report(eval_class_sets['PTP1B_CSP'], yhat))
        pass


    def _supervised_scale_features(self, X_array_train, X_array_eval):
        """Scale all features with either MinMaxScaler or StandardScaler Scaler.
        implementation for supervised and unsupervised is different to prevent
        information leakage."""
        if self.scaling_method == "min_max":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(X_array_train)
        train_data_scaled = scaler.transform(X_array_train)
        eval_data_scaled = scaler.transform(X_array_eval)
        return train_data_scaled, eval_data_scaled


    def _describe_ml_planned(self):
        """Prints out a summary to the user of what machine learning protoctol they have selected."""
        train_pcent = 100 - (self.evaluation_split_ratio*100)
        eval_pcent = self.evaluation_split_ratio*100

        out_text = "\n"
        out_text += "Below is a summary of the machine learning you have planned.\n"
        out_text += f"You will use {self.cross_validation_splits}-fold cross validation and perform {self.cross_validation_repeats} repeats.\n"

        out_text += f"{train_pcent}% of you data will be used for training, which is {len(self.train_data_scaled)} observations.\n"
        out_text += f"{eval_pcent}% of you data will be used for evaluating the best models from cross validation, which is {len(self.eval_data_scaled)} observations.\n"

        out_text += f"You will evaluate the following models and an exhausitive search patten.\n"


        out_text += "If you're happy with the above, lets get model building!"
        return out_text