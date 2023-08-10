# this file is used for defining all the constant variables
# after defining all the variable we will connect all the variable, for connecting all the variable we write code on configuration file
#artifact folder is used to store all the output which are produced by various python file (e.g data_ingestion.py, model_trainer.py, etc.)

#artifacts -> pipeline_folder -> timestamp -> output

#constant file -> configuration file -> data transformation -> utils file -> data transformation

#model training 
# 1. constant -> Done
# 2. configuration -> Done
# 3. model training
# 4. utils file (helper function)

import os, sys
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

#New_ML_Project_with_modular_coding directory 
ROOT_DIR_KEY = os.getcwd()
DATA_DIR = "Notebook"
DATA_DIR_KEY = "food_delivery_dataset.csv"

ARTIFACT_DIR_KEY = "Artifact"

#data ingestion variables
DATA_INGESTION_KEY = 'data_ingestion' # this directory will contain two other directory DATA_INGESTION_RAW_DATA_DIR and DATA_INGESTION_INGESTED_DATA_DIR_KEY

DATA_INGESTION_RAW_DATA_DIR = 'raw_data_dir' #this directory is used to save raw csv file
DATA_INGESTION_INGESTED_DATA_DIR_KEY = 'ingested_dir' #this directory is used to save train csv and test csv files

RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv'


#data transformation variables
DATA_TRANSFORMATION_KEY = 'data_transformation' # this directory will contain two other directory DATA_INGESTION_RAW_DATA_DIR and DATA_INGESTION_INGESTED_DATA_DIR_KEY
DATA_PREPROCESSOR_DIR = 'processor'
DATA_TRANSFORMATION_PROCESSING_OBJ = 'processor.pkl'
DATA_TRANSFORM_DIR = 'transformation'
TRANSFORM_TRAIN_DIR_KEY = 'train.csv'
TRANSFORM_TEST_DIR_KEY = 'test.csv'

#model training variables
MODEL_TRAINER_KEY = 'model_trainer'
MODEL_TRAINER_OBJECT = 'model.pkl'


