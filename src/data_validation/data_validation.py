from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
from src.entity.config_entity import DataValidationConfig,TrainingPipelineConfig
from src.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from scipy.stats import ks_2samp
from src.utils.main_utils.utils import write_yml
import os,sys
import pandas as pd
import numpy as np

class DataValidation:
    def __init__(self,
                 data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_config = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data(self):
        try:
            logging.info("Getting train and test data for validation")
            train_data = pd.read_csv(self.data_ingestion_config.train_file_path)
            test_data = pd.read_csv(self.data_ingestion_config.test_file_path)
            return train_data,test_data
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_drift_report(self,traindata:pd.DataFrame,testdata:pd.DataFrame,threshold:float=0.05):
        try:
            logging.info("Initialising drift report")
            status = False
            report = {}
            for col in traindata.columns:
                d1 = traindata[col]
                d2 = testdata[col] 
                drift = ks_2samp(d1,d2)
                if drift.pvalue>=threshold:
                    drift_found = False
                else:
                    drift_found=True
                    status = True
                
                report.update(
                    {
                        col:{
                            "pvalue":float(drift.pvalue),
                            "status":drift_found
                        }
                    }
                )
            
            drift_report_file_path = self.data_validation_config.drift_report_file
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yml(drift_report_file_path,report)
            logging.info(f"Drift report final status {status}")
            logging.info("Drift report completed")
            return status
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_data,test_data = self.get_data()
            status = self.get_drift_report(train_data,test_data)
            
            if not status:
                dir_path = os.path.join(self.data_validation_config.valid_data_dir)
                os.makedirs(dir_path,exist_ok=True)
                train_data.to_csv(
                    self.data_validation_config.train_valid_data_file_path,index=False,header=True
                )
                test_data.to_csv(
                    self.data_validation_config.test_vaild_data_file_path,index=False,header=True
                )
                valid_train_file_path = self.data_validation_config.train_valid_data_file_path
                valid_test_file_path = self.data_validation_config.test_vaild_data_file_path
                invalid_train_file_path = None
                invalid_test_file_path = None
            else:
                dir_path = os.path.join(self.data_validation_config.invalid_data_dir)
                os.makedirs(dir_path,exist_ok=True)
                train_data.to_csv(
                    self.data_validation_config.train_invalid_data_file_path,index=False,header=True
                )
                test_data.to_csv(
                    self.data_validation_config.test_invalid_data_file_path,index=False,header=True
                )
                valid_train_file_path = None
                valid_test_file_path = None
                invalid_train_file_path = self.data_validation_config.train_invalid_data_file_path
                invalid_test_file_path = self.data_validation_config.test_invalid_data_file_path
                
            data_validation_artifact = DataValidationArtifact(
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=invalid_train_file_path,
                invalid_test_file_path=invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            return data_validation_artifact
                 
        except Exception as e:
            raise CustomException(e,sys)