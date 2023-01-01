import joblib
import numpy as np

model = joblib.load(r'Model/Linear_Regression_Optimum_Model.joblib')

def predict_yield(input_values):
    """
    Predict the class of a given data point.
    """
    return model.predict(input_values)