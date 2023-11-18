import numpy as np
import pandas as pd
import os
import sys

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from exception import CustomException
from logger import logging
from utils import save_file

@dataclass
class DataTransformConfig:
    preprocess_obj_file = os.path.join("artifacts", "preprocess.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformConfig()
    def get_preprocess_data(self):
        try:
            logging.info("Preprocess initiate")
            num_feature = ['current_score','crr','balls_left'	
                        ,'wickets_left'	,'last_five' ]
            cat_feature  = ['batting_team','bowling_team','city']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoder", OneHotEncoder(sparse = False))
                ]
            )

            preprocess = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline,num_feature),
                    ('cat_pipeline', cat_pipeline,cat_feature)
                ]
            )
            logging.info("Preprocess complete")

        except Exception as e:
            raise CustomException(e, sys)
        
        return preprocess
    

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data read succesfuly")

            preprocess_obj = self.get_preprocess_data()
            total_runs = "total_runs"

            y_train = train_df[total_runs]
            x_train = train_df.drop(columns=[total_runs], axis=1)
            y_test = test_df[total_runs]
            x_test = test_df.drop(columns=[total_runs], axis=1)
            logging.info("Data split succesfuly")
            
            logging.info("Preprocessing started")
            x_train_arr = preprocess_obj.fit_transform(x_train)
            x_test_arr = preprocess_obj.transform(x_test)
            logging.info("Preprocessing finish")

            train_arr = np.c_[x_train_arr,np.array(y_train)]
            test_arr = np.c_[x_test_arr, np.array(y_test)]

            save_file(self.data_transform_config.preprocess_obj_file,preprocess_obj)

            logging.info("File save succesfuly")
            logging.info('Transformation completed and returns train_arr, test_arr')
            
            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e,sys)

        
    
        









