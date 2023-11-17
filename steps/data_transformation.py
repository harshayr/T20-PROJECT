import numpy as np
import pandas as pd
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class SplitData:
    def __init__(self,data):
        self.data = data
    def split_data(self):
        train_df,test_df = train_test_split(self.data,shuffle=True,test_size=0.3 )
        return train_df,test_df
    

class TransformData:
    def __init__(self,data):
        self.data = data

    def process_data(self,data):
        split = SplitData(data)
        train_df,test_df = split.split_data()
        num_data_train = train_df.select_dtypes(exclude = "object").columns 
        cat_data_train = train_df.select_dtypes(include = "object").columns 
        num_data_test = test_df.select_dtypes(exclude = "object").columns 
        cat_data_test = test_df.select_dtypes(include = "object").columns

        num_pipeline = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler',StandardScaler())
            ]
        )

        num_pipeline_test =Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy="median"))
            ]
        )

        cat_pipeline = Pipeline(
            steps= [
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ('encoder',OneHotEncoder())   
            ]
        )
        preprocess = ColumnTransformer(
            [
                ("num_pipeline",num_pipeline,num_data_train),
                ("cat_pipeline", cat_pipeline,cat_data_train),
                ("num_pipeline_test",num_pipeline_test,num_data_test),
                ("num_pipeline_test",cat_pipeline,cat_data_test)
            ]
        )

        return preprocess


def transform_df(data):

    df = SplitData(data)
    pre = TransformData(data)
    preprocessor = pre.process_data()
    train_df,test_df = df.split_data()
    x_train = train_df[:,:-1]
    y_train = train_df[:,-1]
    x_test = test_df[:,:-1]
    y_test = test_df[:,-1]

    x_train_arr = preprocessor.fit_transform(x_train)
    x_test_arr = preprocessor.transform(x_test)

    return x_train_arr,y_train,x_test_arr,y_test













