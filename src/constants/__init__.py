import os
import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


TARGET_COLUMN:str = "diagnosis"
ARTIFACT_DIR:str = "Artifacts"
FILE_NAME:str = "breast-cancer.csv"
PIPELINE_NAME:str = "Cancer_Prediction"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

 
'''
Data Ingestion constants
'''
DATA_INGESTION_DIR:str = "data_ingestion"
TRAIN_TEST_SPLIT_RATIO:float  = 0.2
RAW_FILE_PATH:str = "Cancer_Data\\breast-cancer.csv"

'''
Data Validation constant
'''
DATA_VALIDATION_DIR:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "valid_data"
DATA_VALIDATION_INVALID_DIR:str = "invalid_data"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_NAME:str = "report.yml"

'''
Data Transformation constants
'''

DATA_TRANSFORMATION_DIR:str = "data_transformed"
DATA_TRANSFORMATION_FILE_PATH:str = "data"
DATA_TRANSFORMATION_OBJECT_PATH:str = "Preprocessor"
DATA_TRANSFORMATION_OBJECT:str = "preprocessor.pkl"
KNN_IMPUTER_PARAMS = {
    "missing_values":np.nan,
    "n_neighbors":3,
    "weights": "uniform"
}

'''
Model trainer constants
'''

MODEL_TRAINER_DIR_PATH:str = "model_trainer"
MODEL_TRAINER_FILE_PATH:str = "trainer"
MODEL_TRAINER:str = "model.pkl"
MODEL_TRAINER_REPORT_PATH:str = "report"
MODEL_TRAINER_REPORT:str = "report.yml"
MODELS:dict = {
    "Logistic_Regression":LogisticRegression(),
    "Random_Forest":RandomForestClassifier(),
    "XGBoost":XGBClassifier()
}
MODEL_PARAMS: dict = {
    "Logistic_Regression": [  
        {  
            'penalty': ['l2', None],  
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
            'verbose': [0],
            'n_jobs': [-1]
        },
        {  
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'verbose': [0],
            'n_jobs': [-1]
        },
        {  
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['saga'],
            'verbose': [0],
            'n_jobs': [-1]
        }
    ],
    "Random_Forest": {  
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 128, 256],
        'verbose': [0],
        'n_jobs': [-1]
    },
    "XGBoost": {
    'learning_rate': [0.01, 0.05, 0.1],  
    'max_depth': [3, 5, 7],              
    'subsample': [0.8, 0.9],             
    'colsample_bytree': [0.8, 0.9],      
    'gamma': [0, 0.1, 0.2]               
}
}


    
    
    



