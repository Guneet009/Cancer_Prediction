from src.data_ingestion.data_ingestion import DataIngestion
from src.data_validation.data_validation import DataValidation
from src.data_transformation.data_transformation import DataTransformation
from src.model_trainer.model_trainer import ModelTrainer
from src.logger.logging import logging
from src.Custom_Exception.CustomException import CustomException
from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact,DataValidationArtifact,ModelTrainerArtifact
import sys, os


class TrainingPipeline:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating Data Ingestion")
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact:DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed :{data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info("Initiating Data Validation")
            
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact,self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Completed Data Validation :{data_validation_artifact}") 
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("Initiating Data Transformation")
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(self.data_transformation_config,data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed :{data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Initiaing Model Training")
            self.model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(self.model_trainer_config,data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiatte_model_training()
            logging.info(f"Model Training Completed :{model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    
                        
        
    def run_pipeline(self):
        try:
            logging.info("Running Pipeline")
            data_ingestion_artifact = self.initiate_data_ingestion()
            data_validation_artifact = self.initiate_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.initiate_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.initiate_model_trainer(data_transformation_artifact)
           
            logging.info("Pipeline run completed")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)