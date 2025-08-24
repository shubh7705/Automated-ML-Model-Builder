import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

def preProcess(X_train, X_test, y_train, y_test):
    object_columns = X_train.select_dtypes(include='object').columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    transformer = ColumnTransformer(transformers=[
        ('scale', StandardScaler(), numeric_cols),
        ('encode', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), object_columns)
    ], remainder='passthrough')

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    if y_train.dtype == 'O':  # 'O' stands for object
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test


def remove_columns(X_train, X_test, removes):
    X_train = X_train.drop(columns=removes, axis=1)
    X_test = X_test.drop(columns=removes, axis=1)
    return X_train, X_test


def fill_na(data):
    if data[data.duplicated()].shape[0]:
        data = data.drop_duplicates()
    na_cols = data.isna().sum().fillna(0)
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    object_cols = data.select_dtypes(include='object').columns.tolist()
    na_cols = na_cols[na_cols > 0]

    option = st.selectbox(
        "Missing Value Handling:",
        ("Remove Missing Rows", "Fill Missing Values")
    )

    if option == "Remove Missing Rows":
        data = data.dropna()
    else:
        col1,col2 = st.columns(2)
        with col1:
           numeric_option = st.selectbox(
              "Select Method for Numeric Columns",
              ("Mean", "Median", "KNNImputer")
          )

        # Separate numeric and object columns with NaNs
           numeric_na_cols = [col for col in na_cols.index if col in numeric_cols]
           object_na_cols = [col for col in na_cols.index if col in object_cols]

           if numeric_option == "KNNImputer":
                knn_imputer = KNNImputer(n_neighbors=2)
                data[numeric_na_cols] = pd.DataFrame(
                   knn_imputer.fit_transform(data[numeric_na_cols]),
                   columns=numeric_na_cols,
                   index=data.index
                )

           elif numeric_option == "Mean":
                for col in numeric_na_cols:
                    data[col] = data[col].fillna(data[col].mean())

           elif numeric_option == "Median":
                for col in numeric_na_cols:
                    data[col] = data[col].fillna(data[col].median())
        with col2:

           objective_option = st.selectbox(
            "Select Method for Objective Columns",
            ("Most Frequent", "Missing Indicater")
           )

        # Impute object columns
           if objective_option == "Most Frequent":
               if object_na_cols:
                  imputer = SimpleImputer(strategy='most_frequent')
                  data[object_na_cols] = pd.DataFrame(
                      imputer.fit_transform(data[object_na_cols]),
                      columns=object_na_cols,
                      index=data.index
                      )
               else:
                  data = data.fillna("Missing")

    return data

def is_classification_target(y, threshold=20):
    """
    Heuristically determines if target column is classification.
    - Uses type and number of unique values.
    - threshold: max number of unique values for classification.
    """
    if y.dtype == 'object':
        return True
    if y.dtype.name == 'category':
        return True
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y):
        unique_vals = y.nunique()
        if unique_vals <= threshold:
            return True
    return False

