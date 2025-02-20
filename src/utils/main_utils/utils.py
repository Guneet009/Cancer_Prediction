import os
import sys
import zipfile
import pandas as pd
import pickle
import yaml
from src.Custom_Exception.CustomException import CustomException
from src.logger.logging import logging

class DataBase:
    def __init__(self, API_COMMAND, output_folder="Cancer_Data"):
        os.system(API_COMMAND)
        self.name = API_COMMAND.split('/')
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # Create output folder if it doesn't exist
        self.Open()
        os.remove(self.name[1] + ".zip")
    
    def Open(self):
        self.dataset = []
        zip_path = self.name[1] + ".zip"
        
        with zipfile.ZipFile(zip_path, "r") as zip:
            filenames = zip.namelist()
            zip.extractall(self.output_folder)  # Extract files to the specified folder
        
        for filename in filenames:
            file_path = os.path.join(self.output_folder, filename)  # Construct full file path
            self.dataset.append(pd.read_csv(file_path))
        
        self.filenames = filenames

def load_object(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath,"r") as fileobj:
                pickle.load(fileobj)
        else:
            raise FileNotFoundError
    except Exception as e:
        raise CustomException(e,sys)

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        if os.path.exists(filepath):
            with open(filepath,"rb") as fileobj:
                pickle.dump(obj,fileobj)
        else:
            os.makedirs(dir_path)
            with open(filepath,"rb") as fileobj:
                pickle.dump(obj,fileobj)
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
    