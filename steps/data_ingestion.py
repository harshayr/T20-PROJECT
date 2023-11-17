import pandas as pd
from logger import logging
import sys
from exception import CustomException
from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split

@dataclass  # defifnig variable without using constructor
class DataIngestConfig:
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts",'test.csv')
    raw_path = os.path.join("artifacts", "raw.csv")

class IngestData:
    def __init__(self):
        self.ingest_config = DataIngestConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv("/Users/harshalrajput/Desktop/MLOPS_T20/T20-PROJECT/data/t20_cleandata.csv")
            os.makedirs(os.path.dirname(self.ingest_config.train_path), exist_ok=True)
            df.to_csv(self.ingest_config.raw_path)

            train_df,test_df = train_test_split(df,test_size=0.3,shuffle=True)
            train_df.to_csv(self.ingest_config.train_path, index = False,header = True)
            test_df.to_csv(self.ingest_config.test_path, index = False,header = True)
            logging.info("Data ingestion completed")

        except Exception as e:
            raise CustomException(e, sys)
        return (
            self.ingest_config.train_path,
            self.ingest_config.test_path
        )
    

if __name__ == "__main__":
    obj = IngestData()
    obj.initiate_data_ingestion()











        

    