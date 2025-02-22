from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging
from sklearn.metrics import f1_score,precision_score,recall_score
from src.entity.artifact_entity import ModelTrainerMetric
import sys

def model_evaluator(y_train,y_pred):
    try:
        model_f1_score = f1_score(y_train,y_pred)
        model_precision_score = precision_score(y_train,y_pred)
        model_recall_score = recall_score(y_train,y_pred)
        model_trainer_mertric = ModelTrainerMetric(
            f1_score=model_f1_score,
            precision_score= model_precision_score,
            recall_score=model_recall_score
        )
        return model_trainer_mertric
    except Exception as e:
        raise CustomException(e,sys)