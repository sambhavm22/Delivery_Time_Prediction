import os, sys
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.config.configuration import *
from dataclasses import dataclass
from src.entity.config_entity import *

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from src.utils import save_obj, evaluate_model

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("initiating model trainer")
        try:
            logging.info("splitting data into independent and dependent variables")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], 
                                                test_array[:,:-1], test_array[:,-1])
            logging.info("training the data with various models")
            models = {
                "SVR": SVR(),
                "DecisionTree": DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Gradient Boosted Tree Regressor" : GradientBoostingRegressor()
            }
            
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
            print(model_report)

            logging.info("determining best model name and best model score")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")

            logging.info(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")

            save_obj(file_path=self.model_trainer_config.model_trainer_obj_file_path,
                    obj=best_model)
            logging.info("model training completed")
        except Exception as e:
            raise CustomException(e, sys)