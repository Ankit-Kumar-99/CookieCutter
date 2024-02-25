import numpy as np
import pandas as pd


def make_prediction(model,log_reg, X_test):
    #X_test = X_test.reset_index(drop=True)
    prediction = None 
    prediction2 = None
    pjme_test_fcst = None
    print("Making predictions...")
    
    if "{{cookiecutter.ml_task}}" == "classification":

        # model predictions 
        #X_test_for_prediction = X_test.drop('diagnosis', axis=1)
        prediction = model.predict(X_test)
        prediction2 = log_reg.predict(X_test)
        
        print(f"Prediction from KNN: {prediction}")
        print(f"Prediction from Logistic Regression: {prediction2}")


    elif "{{cookiecutter.ml_task}}" == "timeseries":
        print("Making predictions...")
        print(X_test)
        
        X_test_df = pd.DataFrame({'ds': X_test}) 
    
        pjme_test_fcst = model.predict(df=X_test_df)
        # Predict on training set with model
        # pjme_test_fcst = model.predict(df=X_test.reset_index() \
        #                            .rename(columns={'Datetime':'ds'}))
        
        print(f"Prediction: {pjme_test_fcst}")


    else:
        print("Invalid machine learning task.")
    
    return prediction, prediction2, pjme_test_fcst