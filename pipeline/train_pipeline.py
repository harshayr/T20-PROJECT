from exception import CustomException
from logger import logging
from steps.model_train import ModelTrain
from steps.data_ingestion import IngestData
from steps.data_transformation import DataTransformation
from dataclasses import dataclass


class TrainPipeline:

    def __init__(self):

        self.ingestdata = IngestData()
        self.transformdata = DataTransformation()
        self.modeltrain = ModelTrain()

    def training_pipeline(self,data_path):
        train_data_path, test_data_path = self.ingestdata.initiate_data_ingestion(data_path = data_path)
        train_arr, test_arr = self.transformdata.initiate_data_transformation(train_data_path, test_data_path)
        self.modeltrain.model_train(train_arr, test_arr)









