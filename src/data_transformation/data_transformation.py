from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from src.utils.main_utils.utils import save_object,save_object_as_numpy_arr
from src.constants import TARGET_COLUMN,KNN_IMPUTER_PARAMS
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os,sys

class DataTransformation:
    def __init__(self,
                 data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_csv_data(self):
        try:
            train_data = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_data = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            return train_data,test_data
        except Exception as e:
            raise CustomException(e,sys)
    
    def data_transformation_object(self):
        try:
            logging.info("Intializing transformation pipeline")
            imputer = KNNImputer(**KNN_IMPUTER_PARAMS)
            processor:Pipeline = Pipeline([("imputer",imputer)])
            logging.info("Transformation pipeline initialized")
            return processor
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_transformation(self):
        try:
            logging.info("Reading data from validation folder")
            train_set,test_set = self.get_csv_data()
            X_train  = train_set.drop(columns=[TARGET_COLUMN],axis=1)
            y_train = train_set[TARGET_COLUMN].replace({'B':0,'M':1})
            X_test = test_set.drop(columns=[TARGET_COLUMN],axis=1)
            y_test = test_set[TARGET_COLUMN].replace({'B':0,'M':1})
            
            preprocessor = self.data_transformation_object()
            preprocessor_obj = preprocessor.fit(X_train)
            logging.info("Created preprocessor object")
            '''
            No need of transformation since their are  no null value. Found out using EDA
            '''
            # transformed_X_train = preprocessor_obj.transform(X_train)
            # transformed_X_test = preprocessor_obj.transform(X_test)
            logging.info("Saving numpy array")
            train_arr = np.c_[np.array(X_train),np.array(y_train)]
            test_arr = np.c_[np.array(X_test),np.array(y_test)]
            
            save_object_as_numpy_arr(
                self.data_transformation_config.data_transformation_train_file,train_arr
                )
            save_object_as_numpy_arr(
                self.data_transformation_config.data_transformation_test_file,test_arr
                )
            save_object(
                self.data_transformation_config.data_transformation_object,preprocessor_obj
                )
            save_object("final_model/preprocessor.pkl",preprocessor_obj)
            logging.info("Saved numpy arrays and preprocessor objects")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.data_transformation_train_file,
                transformed_test_file_path= self.data_transformation_config.data_transformation_test_file,
                preprocessor_object_path=self.data_transformation_config.data_transformation_object
            )
            return data_transformation_artifact
            
        except Exception as e:
            raise CustomException(e,sys)