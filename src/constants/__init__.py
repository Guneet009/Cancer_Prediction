import os
import numpy as np
import pandas as pd
import sys

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


