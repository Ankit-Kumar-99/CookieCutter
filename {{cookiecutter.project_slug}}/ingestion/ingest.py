import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def ingest_data():
    if "{{cookiecutter.ml_task}}" == "classification":
        print("Loading data...")
        dataset_path = "/home/sigmoid/Downloads/classification.csv"
        data = pd.read_csv(dataset_path)
        print(f"Data loaded from {dataset_path}")
        
    
    elif "{{cookiecutter.ml_task}}" == "timeseries":
        print("Loading data...")
        dataset_path = "/home/sigmoid/Downloads/AEP_hourly.csv"
        data = pd.read_csv(dataset_path)

        

        print(f"Data loaded from {dataset_path}")
    
    else:
        print("Invalid machine learning task.")
        return None, None
    
    return data