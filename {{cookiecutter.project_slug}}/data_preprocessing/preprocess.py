import numpy as np

def preprocess_data(data):
    print("Preprocessing the loaded data...")
    if "{{cookiecutter.ml_task}}" == "classification":
        print(f"The original dataframe has {data.shape[1]} columns.")
        # preprocessing 
        
        data.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
        data['diagnosis'] = data['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)
        
        # removing highly correlated features

        corr_matrix = data.corr().abs() 

        mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
        tri_df = corr_matrix.mask(mask)

        to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]

        data = data.drop(to_drop, axis = 1)

        print(f"The reduced dataframe has {data.shape[1]} columns.")
        
        # Split the data based on the date
        
        split_point = int(len(data) * 0.8)
        current_data = data.iloc[:split_point, :]
        reference_data = data.iloc[split_point:, :]
        
    
    elif "{{cookiecutter.ml_task}}" == "timeseries":
        
        split_point = int(len(data) * 0.8)
        current_data = data.iloc[:split_point, :]
        reference_data = data.iloc[split_point:, :]
    
    else:
        print("Invalid machine learning task.")
        return None, None
    print("Data preprocessed.")
    
    return current_data, reference_data