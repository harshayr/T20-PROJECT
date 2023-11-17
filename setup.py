from setuptools import find_packages, setup  # ussed to find all packages in mlproject
from typing import List
# this contain all the information about project 

hyphone_e = "-e ."  # this is for connect requriment.txt with setup
def get_requirment(file_path:str)->List[str]:
   ''' 
   this functionreturns the list of requirments
   
   '''
   requirments = []
   with open (file_path) as file_obj:
      requirments = file_obj.readlines()# but this will add extra /n for each line 
      requirments = [req.replace("/n",'') for req in requirments]
      if hyphone_e in requirments:
         requirments.remove(hyphone_e)

   return requirments

setup(
    name='mlproject',
    author='Harsha',
    version='0.0.1',
    author_email="rajputharshal2002@gmail.com",
    packages=find_packages(),
    install_requires = get_requirment('requirment.txt')
    )
