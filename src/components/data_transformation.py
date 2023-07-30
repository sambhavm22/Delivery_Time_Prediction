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

from src.utils import save_obj

#FE
class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        logging.info("*************** feature Engineering Started ***************************")

    def distance_numpy(self, df, lat1, lon1, lat2, lon2):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12734 * np.arccos(np.sort(a))

    def transform_data(self, df):
        try:
            
            df.drop(["ID"], axis=1, inplace = True)
            
            logging.info("calling distance numpy function to calculate the distance")  
            self.distance_numpy(df, 'Restaurant_latitude', 
                                'Restaurant_longitude',
                                'Delivery_location_latitude',
                                'Delivery_location_longitude')
            logging.info("dropping unnecessary columns")
            df.drop(['Delivery_person_ID', 'Restaurant_latitude','Restaurant_longitude',
                                'Delivery_location_latitude',
                                'Delivery_location_longitude',
                                'Order_Date','Time_Orderd','Time_Order_picked'], axis=1,inplace=True)

            logging.info("dropping columns from our original dataset")
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def fit(self,X,y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        try:
            transformed_df = self.transform_data(df=X)
            return transformed_df
        except Exception as e:
            raise CustomException(e, sys)

#data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('giving ranking to ordinal categorical features')
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weather_conditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']

            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_columns = ['Road_traffic_density', 'Weather_conditions']
            numerical_columns=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition', 'multiple_deliveries','distance']
            
            logging.info("defining numerical, categorical and ordinal pipeline")
            #numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            ordinal_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density, Weather_conditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_columns)
            ])
            logging.info("pipeline steps completed")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps=[('fe', Feature_Engineering())])

            return feature_engineering
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('obtaining FE object')
            fe_obj = self.get_feature_engineering_object()

            logging.info('fit transform train data and transform test data using preprocessor object')
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)

            train_df.to_csv('train_data.csv')
            test_df.to_csv('test_data.csv')

            preprocessor_obj = self.get_data_transformation_obj()
            target_column_name = 'Time_taken (min)'

            logging.info("splitting train and test data into dependent variable and independent variables")
            X_train = train_df.drop(columns = target_column_name, axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name, axis=1)
            y_test = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training data frame and testing data frame.")
            X_train = preprocessor_obj.fit_transform(X_train)
            X_test = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_file_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_file_path, index = False, header=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_file_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_file_path, index = False, header=True)

            logging.info("saving preprocessing object")
            save_obj(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = fe_obj)

            logging.info("saving feature engineering object")
            save_obj(file_path = self.data_transformation_config.feature_eng_obj_path,
                    obj = fe_obj)
            
            return(train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


