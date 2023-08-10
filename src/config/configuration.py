from src.constants import *
import os, sys

ROOT_DIR = ROOT_DIR_KEY

#data ingestion related variables
DATASET_PATH = os.path.join(ROOT_DIR, DATA_DIR, DATA_DIR_KEY)

RAW_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY, 
                            CURRENT_TIME_STAMP, DATA_INGESTION_RAW_DATA_DIR,
                            RAW_DATA_DIR_KEY)

TRAIN_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                            CURRENT_TIME_STAMP, DATA_INGESTION_INGESTED_DATA_DIR_KEY,
                            TRAIN_DATA_DIR_KEY)

TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                            CURRENT_TIME_STAMP, DATA_INGESTION_INGESTED_DATA_DIR_KEY,
                            TEST_DATA_DIR_KEY)



#data transformation related variables

PREPROCESSOR_OBJ_FILE = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_TRANSFORMATION_KEY,
                                    DATA_PREPROCESSOR_DIR, DATA_TRANSFORMATION_PROCESSING_OBJ)


TRANSFORM_TRAIN_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_TRANSFORMATION_KEY,
                                        DATA_TRANSFORM_DIR, TRANSFORM_TRAIN_DIR_KEY)

TRANSFORM_TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_TRANSFORMATION_KEY,
                                        DATA_TRANSFORM_DIR, TRANSFORM_TEST_DIR_KEY)

FEATURE_ENG_OBJ_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_TRANSFORMATION_KEY,
                                        DATA_PREPROCESSOR_DIR, 'feature_eng.pkl')

#model trainer related variables

MODEL_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, MODEL_TRAINER_KEY, MODEL_TRAINER_OBJECT)


