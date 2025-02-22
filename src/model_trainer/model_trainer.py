from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
from src.constants import MODELS,MODEL_PARAMS
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact,ModelTrainerMetric
from src.utils.main_utils.utils import save_object,load_object_as_numpy_arr,write_yml
from sklearn.model_selection import GridSearchCV
from src.utils.ml_utils.utils import model_evaluator
import warnings
warnings.filterwarnings('ignore')
import os,sys
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self,
                 model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
    
    def get_array(self):
        try:
            logging.info("Loading numpy objects")
            train_arr = load_object_as_numpy_arr(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_object_as_numpy_arr(self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Loaded numpy objects")
            return train_arr,test_arr
        except Exception as e:
            raise CustomException(e,sys)
    
    def model_training(self,X_train,y_train,X_test,y_test,model:dict,model_params:dict):
        try:
            best_model = None
            best_score = -float("inf")
            best_params = None
            report = []
            models = list(model.keys())
            for i in range(len(models)):
                keys = models[i]
                est = model[keys]
                params = model_params[keys]
                gv = GridSearchCV(est,params,cv=5,verbose=0)
                gv.fit(X_train,y_train)
                est.set_params(**gv.best_params_)
                est.fit(X_train,y_train)
                y_pred_train = est.predict(X_train)
                y_pred_test = est.predict(X_test)


                
                train_evaluation = model_evaluator(y_train=y_train, y_pred=y_pred_train)
                test_evaluation = model_evaluator(y_train=y_test, y_pred=y_pred_test)  

                
                model_report = {
                    "Model": keys,
                    "Best Parameters": gv.best_params_,
                    "Train Evaluation": train_evaluation,
                    "Test Evaluation": test_evaluation
                }
                report.append(model_report)

                
            
                if test_evaluation.f1_score> best_score: 
                    best_score = test_evaluation.f1_score
                    best_model = gv.best_estimator_    
            
            logging.info(f"Best model is {best_model}")

    
            write_yml(self.model_trainer_config.model_trainer_report, report)

            return best_model
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiatte_model_training(self):
        try:
            train_arr,test_arr = self.get_array()
            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            best_model =  self.model_training(X_train,y_train,X_test,y_test,MODELS,MODEL_PARAMS)
            # best_model.set_params(**best_params)
            best_model_obj = best_model.fit(X_train,y_train)
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            save_object(self.model_trainer_config.model_trainer_obj,best_model_obj)
            save_object("final_model/model.pkl",best_model_obj)
            train_evaluator:ModelTrainerMetric = model_evaluator(y_train,y_train_pred)
            test_evaluator:ModelTrainerMetric = model_evaluator(y_test,y_test_pred)
            model_train_artifact = ModelTrainerArtifact(
                model_object_path=self.model_trainer_config.model_trainer_obj,
                training_data_artifact=train_evaluator,
                test_data_artifact=test_evaluator
            )
            return model_train_artifact
        except Exception as e:
            raise CustomException(e,sys)    
    