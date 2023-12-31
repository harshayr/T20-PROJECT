import pandas as pd
import sys
from logger import logging
from exception import CustomException
from logger import logging

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

    def initiate_data_ingestion(self,data_path:str):
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv(data_path)
            os.makedirs(os.path.dirname(self.ingest_config.train_path), exist_ok=True)
            df.to_csv(self.ingest_config.raw_path)

            train_df,test_df = train_test_split(df,test_size=0.3,shuffle=True)
            train_df.to_csv(self.ingest_config.train_path, index = False,header = True)
            test_df.to_csv(self.ingest_config.test_path, index = False,header = True)
            logging.info("Data ingestion completed")

            return (
            self.ingest_config.train_path,
            self.ingest_config.test_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
    













        

    