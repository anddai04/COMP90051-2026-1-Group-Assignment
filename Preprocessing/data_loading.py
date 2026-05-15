from pathlib import Path
import pandas as pd

def save_data(df, OUTPUT_FILENAME, OUTPUT_FOLDER, FILE_SUFFIX='.csv'):
    '''
    save data to folder
    '''
    NOTEBOOK_DIRECTORY = Path.cwd()
    DATA_DIRECTORY = (
        NOTEBOOK_DIRECTORY / f"../Data/{OUTPUT_FOLDER}"
    ).resolve()

    OUTPUT_PATH = (
        DATA_DIRECTORY / f"{OUTPUT_FILENAME}{FILE_SUFFIX}"
    )
    df.to_csv(
        OUTPUT_PATH, index=False 
    )

def read_data(INPUT_FILENAME, INPUT_FOLDER, FILE_SUFFIX='.csv'):
    """
    Read the data
    """
    NOTEBOOK_DIRECTORY = Path.cwd()

    DATA_DIRECTORY = (
        NOTEBOOK_DIRECTORY / f"../Data/{INPUT_FOLDER}"
    ).resolve()

    INPUT_PATH = (
        DATA_DIRECTORY / f"{INPUT_FILENAME}{FILE_SUFFIX}"
    )
    return pd.read_csv(INPUT_PATH)

def train_test_pipeline(df_train, df_test, functions_list):
    '''
    Apply functions to both train and test set
    '''
    for func in functions_list:
        df_train, df_test = func(df_train, df_test)
    return df_train, df_test