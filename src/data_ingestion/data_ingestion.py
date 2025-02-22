from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
from src.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from src.constants import RAW_FILE_PATH
import sys,os
import pandas as pd
import numpy as np

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_as_dataframe(self)->pd.DataFrame:
        try:
            logging.info("Reading Data")
            df = pd.read_csv(RAW_FILE_PATH)
            df.replace({"na":np.nan},inplace=True)
            df.drop(columns=['id'],axis=1,inplace=True)
            logging.info("Data read successfully")
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def split_data(self,dataframe:pd.DataFrame):
        try:
            logging.info("Initiating train test split")
            train_data,test_data = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_ratio,random_state=23)
            logging.info("Train test split completed")
            return train_data,test_data
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating Data Ingestion")
            # df = self.get_data_as_dataframe()
            # train_set, test_set = train_test_split(df,test_size=self.data_ingestion_config)
            df = self.get_data_as_dataframe()
            train_data,test_data = self.split_data(df)
            dir_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(dir_path,exist_ok=True)
            train_data.to_csv(
                self.data_ingestion_config.data_ingestion_train_file_path,index=False,header=True
            )
            test_data.to_csv(
                self.data_ingestion_config.data_ingestion_test_file_path,index=False,header=True
            )
            logging.info("train and test file saved")
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.data_ingestion_train_file_path,
                test_file_path=self.data_ingestion_config.data_ingestion_test_file_path
            )
            
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
            
    
