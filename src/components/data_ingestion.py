from src.components.model_trainer import ModelTrainer
from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.entity.config_entity import *

from src.components.data_transformation import DataTransformation

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("initiating data ingestion")
            df = pd.read_csv(DATASET_PATH)

            logging.info("making artifact directory and saving raw data")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("splitting raw data into train and test dataset")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            logging.info("creating train and test file and saving it")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))