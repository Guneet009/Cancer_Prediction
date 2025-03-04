import os
import sys
import zipfile
import pandas as pd
import numpy as np
import pickle
import yaml
from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging

class DataBase:
    def __init__(self, API_COMMAND, output_folder="Cancer_Data"):
        os.system(API_COMMAND)
        self.name = API_COMMAND.split('/')
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  
        self.Open()
        os.remove(self.name[1] + ".zip")
    
    def Open(self):
        self.dataset = []
        zip_path = self.name[1] + ".zip"
        
        with zipfile.ZipFile(zip_path, "r") as zip:
            filenames = zip.namelist()
            zip.extractall(self.output_folder)  
        
        for filename in filenames:
            file_path = os.path.join(self.output_folder, filename)  
            self.dataset.append(pd.read_csv(file_path))
        
        self.filenames = filenames

def load_object(filepath):
    try:
        if not os.path.exists(filepath):
            raise Exception(f"The file: {filepath} is not exists")
        with open(filepath, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(filepath,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def write_yml(filepath,obj):
    dir_path = os.path.dirname(filepath)
    if os.path.exists(filepath):
        with open(filepath,"w") as fileobj:
            yaml.dump(obj,fileobj)
    else:
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,"w") as file:
            yaml.dump(obj,file)

def save_object_as_numpy_arr(filepath,obj):
    dir_path = os.path.dirname(filepath)
    try: 
        if os.path.exists(dir_path):
            with open(filepath,"wb") as file:
                np.save(file,obj)
        else:
            os.makedirs(dir_path)
            with open(filepath,"wb") as file:
                np.save(file,obj)
        
    except Exception as e:
        raise CustomException(e,sys)

def load_object_as_numpy_arr(filepath):
    try:
        with open(filepath,"rb") as file:
            return np.load(file)
    except Exception as e:
        raise CustomException(e,sys)
    