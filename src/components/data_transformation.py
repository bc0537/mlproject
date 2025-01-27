import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # building pipeline
from sklearn.impute import SimpleImputer # handle  missing value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessing.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transormation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_cols=["writing_score","reading_score"]
            categorical_cols=['gender', 
                              'race_ethnicity', 
                              'parental_level_of_education', 
                              'lunch', 
                              'test_preparation_course'
                            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("StandardScaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("Inputer",SimpleImputer(strategy="most_frequent")), # handle missing value
                    ("OneHotEncoder",OneHotEncoder()), # handle categorical data
                    ("StandardScaler",StandardScaler(with_mean=False)) # handle numerical data

                ]
            )
            logging.info("Numerical columns Scaling completed")
            
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_cols),
                    ("categorical_pipeline",cat_pipeline,categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
            
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed sucessfully")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformation_object()

            target_column_name="math_score"
            numerical_cols=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df) # concatenating target column to train array
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df) # concatenating target column to test array
            ]

            logging.info("Saved Preprocessing objects")

            save_object(

                file_path=self.data_transormation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            
            
            return (
                train_arr,
                test_arr,
                self.data_transormation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)


        

