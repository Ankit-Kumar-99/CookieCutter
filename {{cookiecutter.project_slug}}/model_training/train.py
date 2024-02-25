import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from fbprophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
log_reg = None
X_test = None
y_test = None


def train_model(current_data,reference_data):
    if "{{cookiecutter.ml_task}}" == "classification":
        print("Training a KNN model...")
        print(f"current data \n {current_data}")
        print(f"reference data \n {reference_data}")

        # creating features and label 

        X = current_data.drop('diagnosis', axis = 1)
        y = current_data['diagnosis']

        # splitting data into training and test set


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
        print(f"X_train \n {X_train}")
        print(f"X_test \n {X_test}")    
        print(f"y_train \n {y_train}")  
        print(f"y_test \n {y_test}")

        
        
        model = KNeighborsClassifier()
    
        model.fit(X_train, y_train)
        # Print feature names used during training
        print("Feature names during training:", X_train.columns.tolist())
        print("KNN Model trained successfully.")


        # ####################

 


        print("Comparin Classificaiton Models using Evidently")
        train_probas = pd.DataFrame(model.predict_proba(current_data.drop('diagnosis', axis=1)))
        train_probas.columns = ['no', 'yes']
        test_probas = pd.DataFrame(model.predict_proba(reference_data.drop('diagnosis', axis = 1)))
        test_probas.columns = ['no', 'yes']   


        print(f"train_probas \n {train_probas}")
        
        print(f"test_probas \n {test_probas}")
     

        merged_df = pd.concat([current_data.drop('diagnosis', axis = 1), current_data['diagnosis'], train_probas], axis=1)
        print(f"merged_df \n {merged_df}")
        # Drop rows with NaN values
        cleaned_trained_df = merged_df.dropna()

        # Display the cleaned DataFrame
        print(cleaned_trained_df)

        # Replace values in the "diagnosis" column
        cleaned_trained_df['diagnosis'] = cleaned_trained_df['diagnosis'].map({1.0: 'yes', 0.0: 'no'})

        # Display the DataFrame after mapping the values
        print(f"cleaned trained df {cleaned_trained_df}")
        test_probas.index = range(455, 455 + len(test_probas))

        merged_test_df = pd.concat([reference_data.drop('diagnosis', axis = 1), reference_data['diagnosis'], test_probas], axis=1)
        print(f"merged_test_df \n {merged_test_df}")
        # Drop rows with NaN values
        cleaned_test_df = merged_test_df.dropna()

        # Display the cleaned DataFrame
        print(cleaned_test_df)

        # Replace values in the "diagnosis" column
        cleaned_test_df['diagnosis'] = cleaned_test_df['diagnosis'].map({1.0: 'yes', 0.0: 'no'})

        # Display the DataFrame after mapping the values
        print(f"cleaned test df {cleaned_test_df}")

        column_mapping = ColumnMapping()

        column_mapping.target = 'diagnosis'
        column_mapping.prediction = ['yes', 'no']
        column_mapping.pos_label = 'yes'


        classification_performance_report = Report(metrics=[
        ClassificationPreset(),
        ])

        classification_performance_report.run(reference_data=cleaned_trained_df, current_data=cleaned_test_df, column_mapping = column_mapping)

    
        classification_performance_report.save_html('performance.html')
        print("Classification performance report saved to performance.html")

        ####

        print("training for logistic regression")
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        print("Logistic Regression Model trained successfully.") 

        return model,log_reg

        

    elif "{{cookiecutter.ml_task}}" == "timeseries":
        #Prophet model expects the dataset to be named a specific way. We will rename our dataframe columns before feeding it into the model.
        print("Training a PROPHET model...")
        print(current_data.columns)
        current_data.reset_index() \
            .rename(columns={'Datetime':'ds',
                     'AEP_MW':'y'}).head()
        

        # Setup and train model and fit
        model = Prophet()
        model.fit(current_data.reset_index() \
              .rename(columns={'Datetime':'ds',
                               'AEP_MW':'y'}))
        print("Model trained successfully.")

        return model,None



    else:
        print("Invalid machine learning task.")
        return None
    
   