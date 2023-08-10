from src.constants import *
from src.config.configuration import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = PREPROCESSOR_OBJ_FILE
    transform_train_file_path:str = TRANSFORM_TRAIN_FILE_PATH
    transform_test_file_path:str = TRANSFORM_TEST_FILE_PATH
    feature_eng_obj_path:str = FEATURE_ENG_OBJ_FILE_PATH

@dataclass
class ModelTrainerConfig:
    model_trainer_obj_file_path:str = MODEL_FILE_PATH   