from pipeline.train_pipeline import TrainPipeline
from exception import CustomException
import sys


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.training_pipeline('/Users/harshalrajput/Desktop/mlops_T20_project/data/t20_cleandata.csv')

