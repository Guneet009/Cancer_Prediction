from src.utils.main_utils.utils import DataBase
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_validation.data_validation import DataValidation
from src.data_transformation.data_transformation import DataTransformation
from src.model_trainer.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact
from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
import warnings
warnings.filterwarnings('ignore')

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
        data_validation_artifacts = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifacts)
        logging.info("Starting data transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config,data_validation_artifacts)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation complete")
        print(data_transformation_artifact)
        logging.info("Intiating training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config,data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiatte_model_training()
        logging.info("model training completed")
        print(model_trainer_artifact)
        
    except Exception as e:
        raise CustomException(e,sys)