import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self,train_array,test_array):
    try:
      logging.info("split training and test input data")

      X_train,y_train,X_test,y_test = (
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )

      models={
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "KNeighbors Regressor": KNeighborsRegressor(),
        "XGBoost Regressor": XGBRegressor(),
        "CatBoost Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor()
      }

      params = {
          "Random Forest": {
              "n_estimators": [8,16,32,64,128,256],  # Number of boosting stages
              
          },
          "Decision Tree": {
              "criterion": ["squared_error", "friedman_mse", "absolute_error","poisson"],  # Function to measure the quality of a split
          },
          "Gradient Boosting": {
              "n_estimators": [8,16,32,64,128,256],  # Number of boosting stages
              "learning_rate": [0.1,0.01,0.05,0.001],  # Step size for updates
              "subsample": [0.6,0.7,0.75,0.8,0.85,0.9]           # Fraction of samples used for fitting
          },
          "Linear Regression": {},
          "KNeighbors Regressor": {
              "n_neighbors": [5, 7, 9, 11],     # Number of neighbors
              # "weights": ["uniform", "distance"],  # Weight function used in prediction
          },
          "XGBoost Regressor": {
              "n_estimators": [8,16,32,64,128,256],  # Number of boosting rounds
              "learning_rate": [0.1,0.01,0.05,0.001],  # Step size shrinkage
          },
          "CatBoost Regressor": {
              "iterations": [30,50,100],     # Number of boosting iterations
              "learning_rate": [0.01, 0.05, 0.1],  # Learning rate
              "depth": [6,8,10],               # Depth of trees
          },
          "AdaBoost Regressor": {
              "n_estimators": [8,16,32,64,128,256],   # Number of boosting stages
              "learning_rate": [0.1,0.01,0.5,0.001],  # Weight applied to each regressor
              # "loss": ["linear", "square", "exponential"]  # Loss function for weight updates
          }
      }

      model_report:dict= evaluate_model(X_train= X_train, y_train =y_train, X_test=X_test, y_test=y_test, models=models,param=params)

      ## to get the best model score from the model report(dictionary)
      best_model_score = max(sorted(model_report.values()))

      # To get the best model name from the model report
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
        ]
      
      best_model = models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException("No best model found")
      
      logging.info("Best found model on both training and testing dataset")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      predicted = best_model.predict(X_test)
      R2_score = r2_score(y_test,predicted)
      return R2_score

    except Exception as e:
      raise CustomException(e, sys)