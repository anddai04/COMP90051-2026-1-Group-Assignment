#!/usr/bin/env python
# coding: utf-8

# ## Set Up Environment

# #### Imports

# In[14]:


# import libraries
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# import dependencies
from data_loading import *


# #### Load Clean Folds

# In[15]:



# In[16]:


# #### Configuration for Reduction

# In[17]:


# medication columns
MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

# purity threshold for concern
PURITY_THRESHOLD = 98
DROP_THRESHOLD = 100


# #### Explore Feature Label Imbalance

# In[18]:


def calculate_column_purity(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    compute purity, defined as percent of values taken up by modal label
    """
    purity_results = []
    total_rows = len(dataset)
    for column in dataset.columns:
        value_counts = dataset[column].value_counts()

        # determine model value and count
        most_common_value = value_counts.index[0]
        most_common_count = value_counts.iloc[0]

        # calculate percentage of values taken up by mode
        purity_percentage = (
            most_common_count / total_rows
        ) * 100

        # save results
        purity_results.append(
            {
                "column": column,
                "most_common_value": most_common_value,
                "most_common_count": most_common_count,
                "purity_percentage": purity_percentage,
                "unique_values": dataset[column].nunique(),
            }
        )

    # aggregate results and sort
    purity_summary = pd.DataFrame(
        purity_results
    ).sort_values(
        "purity_percentage",
        ascending=False,
    )

    return purity_summary.reset_index(drop=True)

# calculate purity
#column_purity_summary = calculate_column_purity(df_train)
#display(column_purity_summary.query(f'purity_percentage >= {PURITY_THRESHOLD}'))


# In[19]:


# break down by medication
#column_purity_summary.loc[column_purity_summary['column'].isin(MEDICATION_COLUMNS)]


# #### Drop Pure Labelled Columns

# In[20]:


# extend to folds
def full_reduce_df(df_train, df_test):
    # calculate purity
    column_purity_summary = calculate_column_purity(df_train)
    pure_columns = list(column_purity_summary
                    .loc[column_purity_summary['purity_percentage']
                         >=DROP_THRESHOLD]
                    ['column'].values)
    df_train = df_train.drop(columns=pure_columns)
    df_test = df_test.drop(columns=pure_columns)
    return df_train, df_test


# ## Feature Encoding and Mapping

# #### Configuration for Encodings

# In[21]:


# ids to map to look up table
LOOKUP_COLUMNS = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

# admission type id map
ADMISSION_TYPE_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped"
}

# discharge disposition id map
DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Transferred to another short term hospital",
    3: "Transferred to skilled nursing facility",
    4: "Transferred to intermediate care facility",
    5: "Transferred to another inpatient institution",
    6: "Discharged to home with home health service",
    7: "Left against medical advice",
    8: "Home under care of IV provider",
    9: "Admitted as inpatient to this hospital",
    10: "Neonate discharged to another hospital",
    11: "Expired",
    12: "Still patient or expected to return",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Transferred within institution",
    16: "Transferred outpatient",
    17: "Transferred rehab facility",
    18: "NULL",
    19: "Expired at home",
    20: "Expired in medical facility",
    21: "Expired, unknown place",
    22: "Transferred psychiatric hospital",
    23: "Transferred long-term care",
    24: "Medicaid only hospice",
    25: "Not Mapped",
    26: "Unknown/Invalid"
}

# admission source id map
ADMISSION_SOURCE_MAP = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from hospital",
    5: "Transfer from skilled nursing facility",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    18: "Transfer from another home health agency",
    19: "Readmission to same home health agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpatient/same facility",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from ambulatory surgery center",
    26: "Transfer from hospice"
}

'''
# omitted as descriptors too long
df["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE_MAP)
df["discharge_disposition"] = df["discharge_disposition_id"].map(DISCHARGE_DISPOSITION_MAP)
df["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE_MAP)

# also in IDS_mapping.csv
# https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
'''


# In[22]:


# ordinal columns and mappings
ORDINAL_COLUMNS = ['age', 'weight']
AGE_COLUMN = 'age'
WEIGHT_COLUMN = 'weight'
UNKNOWN = 'Unknown'

# for age encode mid point
AGE_MAPPING = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

# for weight encode mid point
# end points likely limited
WEIGHT_MAPPING = {
    "[0-25)": 12.5,
    "[25-50)": 37.5,
    "[50-75)": 62.5,
    "[75-100)": 87.5,
    "[100-125)": 112.5,
    "[125-150)": 137.5,
    "[150-175)": 162.5,
    "[175-200)": 187.5,
    ">200": 212.5, # project linearly
    UNKNOWN: UNKNOWN, # keep untouched since too common
}


# In[23]:


# nominal columns to flatten
ONE_HOT_COLUMNS = [
    "race",
    "gender",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "payer_code",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed"
] + MEDICATION_COLUMNS

# decide which columns to one hot encode
# df_reduced.nunique()


# #### Ordinal Encoding

# In[24]:


def ordinal_encode(df_reduced):
    ''' 
    create ordinally encoded numeric features using mid points
    '''
    df_reduced["age"] = (
        df_reduced[AGE_COLUMN]
        .map(AGE_MAPPING)
    )
    df_reduced["weight"] = (
        df_reduced[WEIGHT_COLUMN]
        .map(WEIGHT_MAPPING)
    )
    return df_reduced

def full_ordinal_encode(df_train, df_test):
    '''
    apply ordinal encoding to both train and test
    '''
    return ordinal_encode(df_train), ordinal_encode(df_test)


# #### One Hot Encoding

# In[25]:


def one_hot_encode_train(df_train):
    """
    Fit one-hot encoding structure on training data.
    """
    # ensure columns actually exist
    safe_one_hot_columns = [
        column
        for column in ONE_HOT_COLUMNS
        if column in df_train.columns
    ]

    # encode
    df_train_encoded = pd.get_dummies(
        df_train,
        columns=safe_one_hot_columns,
    )

    # extract the encoded columns
    # ASSUME NO BOOLEAN COLUMNS ORIGINALLY
    bool_columns = (
        df_train_encoded
        .select_dtypes(include="bool")
        .columns
    )

    df_train_encoded[bool_columns] = (
        df_train_encoded[bool_columns]
        .astype(int)
    )

    training_columns = df_train_encoded.columns.tolist()
    return df_train_encoded, training_columns

def one_hot_encode_test(df_test, training_columns):
    """
    Apply training one-hot structure to test data.
    """
    safe_one_hot_columns = [
        column
        for column in ONE_HOT_COLUMNS
        if column in df_test.columns
    ]

    df_test_encoded = pd.get_dummies(
        df_test,
        columns=safe_one_hot_columns,
    )

    bool_columns = (
        df_test_encoded
        .select_dtypes(include="bool")
        .columns
    )

    df_test_encoded[bool_columns] = (
        df_test_encoded[bool_columns]
        .astype(int)
    )

    # Add missing columns from training set.
    df_test_encoded = df_test_encoded.reindex(
        columns=training_columns,
        fill_value=0,
    )

    return df_test_encoded

def full_one_hot_encode(df_train, df_test):
    '''
    Aggregate one hot encoding
    '''
    df_train_encoded, training_columns = one_hot_encode_train(df_train)
    df_test_encoded = one_hot_encode_test(df_test, training_columns)
    return df_train_encoded, df_test_encoded


# ## Aggregate Encodings Pipeline

# In[26]:
