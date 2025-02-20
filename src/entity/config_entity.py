from src import constants
import os,sys
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = constants.PIPELINE_NAME
        self.artifact_name = constants.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        
        


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_INGESTION_DIR
            )
        self.data_ingestion_train_file_path:str = os.path.join(
            self.data_ingestion_dir,
            constants.TRAIN_FILE_NAME
            )
        self.data_ingestion_test_file_path:str = os.path.join(
            self.data_ingestion_dir,
            constants.TEST_FILE_NAME
            )
        self.train_test_ratio:float = constants.TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:
    def __init__(self,train_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            train_pipeline_config.artifact_dir,constants.DATA_VALIDATION_DIR
        )
        self.valid_data_dir:str = os.path.join(
            self.data_validation_dir,constants.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir = os.path.join(
            self.data_validation_dir,constants.DATA_VALIDATION_INVALID_DIR
        )
        self.train_valid_data_file_path = os.path.join(
            self.valid_data_dir,constants.TRAIN_FILE_NAME
        )
        self.test_vaild_data_file_path = os.path.join(
            self.valid_data_dir,constants.TEST_FILE_NAME
        )
        self.train_invalid_data_file_path = os.path.join(
            self.invalid_data_dir,constants.TRAIN_FILE_NAME
        )
        self.test_invalid_data_file_path = os.path.join(
            self.invalid_data_dir,constants.TEST_FILE_NAME
        )
        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,constants.DATA_VALIDATION_DRIFT_REPORT_DIR
        )
        self.drift_report_file = os.path.join(
            self.drift_report_file_path,constants.DATA_VALIDATION_DRIFT_REPORT_NAME
        )
        