from src.utils.main_utils.utils import DataBase
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_validation.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
import sys

if __name__ =="__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data Ingestion completed")
        logging.info("Initialising data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_validation_config,data_ingestion_artifact)
        data_artifacts = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_artifacts)
        
    except Exception as e:
        raise CustomException(e,sys)