#!/usr/bin/env python3
# import necessary libraries for Data Ingestion 
import os
import sys
# from src.exception import CustomException
from ..exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig 




@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifact","train.csv")
    test_data_path:str=os.path.join("artifact","test.csv")
    raw_data_path:str=os.path.join("artifact","data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def intiate_data_ingestion(self):
        logging.info("Data Ingestion Method")

        try:
            df=pd.read_csv("notebook/stud.csv")
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split executed")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("train.csv file created in current directory")
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("test.csv file created in current directory")

            logging.info("Data Ingestion Completed Sucessfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                 
                   )


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.intiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()

    print(model_trainer.initiate_model_trainer(train_array,test_array))





    

