from setuptools import setup,find_packages
from typing import List
from src.Custom_Exception import CustomException
import sys

def get_requirements()->List[str]:
    requirement_list:List[str] = []
    try:
        with open('requirement.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    
        print(sys.path)
        sys.path.insert(0, 'D:\Cancer_Prediction\src\logger\logging.py')
    except Exception as e:
        print(e)
    return requirement_list
    
setup(
    name = "Cancer_Prediction",
    version = "0.0.1",
    author="Guneet Singh",
    packages=find_packages(),
    install_requires = get_requirements()
)
             
