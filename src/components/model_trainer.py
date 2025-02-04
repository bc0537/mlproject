import os
import sys
from dataclasses import dataclass

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("Splitting training and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            logging.info("Model Training")

            models={
                "Random Forest Regressor":RandomForestRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "XGBoost Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "Linear Regression":LinearRegression(),
                "AdaBoost Regressor":AdaBoostRegressor(),
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_score=max(sorted(model_report.values()))


            # to get bets model name from dict

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted_model=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted_model)
            logging.info(f"R2 score of best model is {r2_score}")

            return r2_square


        except Exception as e:
            raise CustomException(e,sys)




