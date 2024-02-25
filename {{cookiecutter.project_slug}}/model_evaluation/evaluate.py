from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error, accuracy_score
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

def evaluate_model(prediction, prediction2 , model, log_reg , X_test, y_test,pjme_test_fcst=None,reference_data=None):
    #X_test = X_test.reset_index(drop=True)
    print("Evaluating model...")
    if "{{cookiecutter.ml_task}}" == "classification":

        # print("Unique values in y_test:", y_test.unique())
        # y_test_binary = (y_test > 0.5).astype(int)



        # # Exclude 'diagnosis' column from X_test
        # X_test_for_prediction = X_test.drop('diagnosis', axis=1)

        # Print feature names during evaluation
        print("Feature names during evaluation:", X_test.columns.tolist())

        # accuracy score
        from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


        knn_acc = accuracy_score(y_test, model.predict(X_test))
        print("KNN The model accuracy is " , knn_acc*100 , "%")
        print("Confusion Matrix for KNN")
        print(confusion_matrix(y_test, prediction))
        print("Classification Report for KNN")
        print(classification_report(y_test, prediction))

        # accuracy score


        #print(accuracy_score(y_train, log_reg.predict(X_train)))

        log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
        print("Logistic Regression Model accuracy is " , log_reg_acc*100 , "%")
        print("Confusion Matrix for Logistic Regression Model")
        print(confusion_matrix(y_test, prediction2))
        print("Classification Report for Logistic Regression Model")
        print(classification_report(y_test, prediction2))
        # KNN_merged_test = pd.concat([y_test,prediction], axis = 1)
        # LR_merged_test = pd.concat([y_test,prediction2], axis = 1)
        # classification_performance_report = Report(metrics=[
        #     ClassificationPreset(),
        # ])

        # classification_performance_report.run(reference_data=KNN_merged_test, current_data=LR_merged_test, column_mapping = column_mapping)

        # classification_performance_report.save_html('classification_performance_report.html')

    elif "{{cookiecutter.ml_task}}" == "timeseries":
        print("Evaluating model...")

        print(reference_data.columns)
        print(pjme_test_fcst.columns)
        mse = mean_squared_error(y_true=reference_data['AEP_MW'],
                   y_pred=pjme_test_fcst['yhat'])



        # Print mean squared error
        print(f"Mean Squared Error: {mse}")

        mae = mean_absolute_error(y_true=reference_data['AEP_MW'],
                   y_pred=pjme_test_fcst['yhat'])
        print(f"Mean Absolute Error: {mae}")



        def mean_absolute_percentage_error(y_true, y_pred): 
        #Calculates MAPE given y_true and y_pred"""
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"Mean absolute percentage error: {mean_absolute_percentage_error(y_true=reference_data['AEP_MW'],y_pred=pjme_test_fcst['yhat'])}")



        print("Model evaluated successfully.")
    else:
        print("Invalid machine learning task.")