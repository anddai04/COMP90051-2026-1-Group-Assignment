from pathlib import Path
import pandas as pd

def save_data(df, OUTPUT_FILENAME, OUTPUT_FOLDER, FILE_SUFFIX='.csv'):
    '''
    save data to folder
    '''
    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "Data" / OUTPUT_FOLDER
    output_path = data_directory / f"{OUTPUT_FILENAME}{FILE_SUFFIX}"
    df.to_csv(
        output_path, index=False 
    )

def read_data(INPUT_FILENAME, INPUT_FOLDER, FILE_SUFFIX='.csv'):
    """
    Read the data
    """
    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "Data" / INPUT_FOLDER
    input_path = data_directory / f"{INPUT_FILENAME}{FILE_SUFFIX}"
    return pd.read_csv(input_path)

def train_test_pipeline(df_train, df_test, functions_list):
    '''
    Apply functions to both train and test set
    '''
    for func in functions_list:
        df_train, df_test = func(df_train, df_test)
    return df_train, df_test