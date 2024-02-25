from ingestion.ingest import ingest_data
from input_data_quality_and_drift_checks.checks import check_data_quality, check_data_drift
from model_training.train import train_model
from model_evaluation.evaluate import evaluate_model
from prediction.predict import make_prediction
from prediction_drift.check_drift import check_prediction_drift
from data_preprocessing.preprocess import preprocess_data

def main():
    data = ingest_data()
    current_data, reference_data = preprocess_data(data)
    check_data_quality(reference_data, current_data)
    check_data_drift(reference_data, current_data)
    model,log_reg = train_model(current_data,reference_data)

    print(f"columns in reference data: {reference_data.columns}")
    if "{{cookiecutter.ml_task}}" == "classification":
        X_test = reference_data.iloc[:, 1:]
        y_test = reference_data.iloc[:, 0].values

    elif "{{cookiecutter.ml_task}}" == "timeseries":
        X_test = reference_data.iloc[:, 0].values
        y_test = reference_data.iloc[:, 1].values
    #print(f"Xtest has these columns {X_test.columns}")

    

    prediction , prediction2 , pjme_test_fcst = make_prediction(model,log_reg, X_test)
    evaluate_model(prediction , prediction2 , model, log_reg,X_test, y_test, pjme_test_fcst, reference_data)
    
    #check_prediction_drift(prediction)

if __name__ == "__main__":
    main()