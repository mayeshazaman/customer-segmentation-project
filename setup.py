from setuptools import setup, find_packages
from typing import List
HYPHEN_E_DOT = "-e ."

def get_requirmentss(file_path:str)-> List[str]:
    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace("\n", "") for req in requirments]
    

    if HYPHEN_E_DOT in requirments:
        requirments.remove(HYPHEN_E_DOT)
    
    return requirments
        
setup(
    name="mlproject",
    version="0.0.1",
    author="niloy",
    author_email="shahriaralam763@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirmentss("requirements.txt")
)