import sys
import os

from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from exception import CustomException
from logger import logging
from utils import save_file
from utils import evaluate_model


@dataclass
class ModelConfig:
    train_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrain:
    def __init__(self) -> None:
        self.train_model_file_path = ModelConfig()
    def model_train(self, train_arr,test_arr):
        try:
            logging.info("Divinding data into x_train,y_train,x_test,y_tes")
            x_train,y_train,x_test,y_test = (train_arr[:,:-1],
                                             train_arr[:,-1],
                                             test_arr[:,:-1],
                                             test_arr[:,-1])
            logging.info("Data divide succesfuly")

            models = {
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "K-Neighbors Classifier": KNeighborsRegressor()
                }
            
            params = {
                    "Decision Tree": {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                    "Random Forest": {
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                        # 'max_features':['sqrt','log2',None],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },

                    "Linear Regression": {},

                    "XGBRegressor": {
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "CatBoosting Regressor": {
                        'depth': [6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost Regressor": {
                        'learning_rate': [.1, .01, 0.5, .001],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    }

                }
            logging.info("Model trainig started")
            modelreport:dict = evaluate_model(x_train,y_train,x_test,y_test,models,params)
            best_score_model = max(list(modelreport.values()))
            best_model_name = list(modelreport.keys())[list(modelreport.values()).index(best_score_model)]
            best_model = models[best_model_name]
            logging.info("Model trainig finish")
            logging.info(f"Model with high accuracy: {best_model_name}")

            if best_score_model<0.6:
                raise CustomException("No best model found")
            
            save_file(
                file_path=self.train_model_file_path.train_model_file_path,
                obj=best_model
                )
            logging.info ("Model save succesfuly")
            
            pred =best_model.predict(x_test)
            r2 = r2_score(y_test,pred)
            
            return r2
        except Exception as e:
            raise CustomException(e,sys)


    

