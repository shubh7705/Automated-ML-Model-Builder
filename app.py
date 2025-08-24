import streamlit as st
import pandas as pd
import model
import preprocessing
import eda
import pickle

st.set_page_config(page_title="ML Workflow App", layout="wide")

# Sidebar: File Upload
with st.sidebar:
    st.title('📂 Upload Your CSV File')
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Main App
st.title("🔍 Machine Learning Workflow")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File successfully uploaded!")
    
    # Show basic data preview
    with st.expander("📄 Preview Data"):
        st.dataframe(data.head())

    # Perform EDA
    with st.expander("📊 Perform EDA"):
        if st.button('Run EDA'):
            eda.data_info(data)

    if model.na_values(data)!=0:
            data = preprocessing.fill_na(data)
            

    # Target Column Selection
    st.subheader("🎯 Select Target and Features")
    column_name = list(data.columns)
    col1,col2 = st.columns(2)
    with col1:
       target_column = st.selectbox("Select column to predict (Target):", column_name)

    if target_column:
        X = data.drop(columns=target_column)
        y = data[target_column]
        dt = y.dtype
        if preprocessing.is_classification_target(y,20):
            dt = 'O'

        # Optional Feature Removal
        with col2:
           removes = st.multiselect("Select columns to remove from features:", list(X.columns))
        # model_option = []
        # if dt == 'O':
        #    model_option = st.selectbox(
        #     "Choose Model:",
        #     ("Logistic Regression","Decision Tree Classifier","Random Forest Classifier","K-Nearest Neighbors (KNN)")
        # )
        # else:
        #     model_option = st.selectbox(
        #         "Choose Model:",
        #         ("Linear Regression","Decision Tree Regressor","Random Forest Regressor","ElasticNet Regression")
        #     )
             

        if st.button("🚀 Train and Predict"):
            # Train-Test Split
            X_train, X_test, y_train, y_test = model.split(X, y)
            
            # Preprocessing
            X_train,X_test = preprocessing.remove_columns(X_train,X_test,removes)

            X_train, X_test, y_train, y_test = preprocessing.preProcess(
                X_train, X_test, y_train, y_test
            )

            # Model Training + Evaluation
            model.model_fitting(X_train, X_test, y_train, y_test, dt)

            # pkl_data = pickle.dumps(model)

            # st.download_button(
            #       label="Download Model",
            #       data=pkl_data,
            #       file_name="model.pkl",
            #       mime="application/octet-stream"
            # )

            # st.subheader("📈 Model Performance")
            # st.write(f"Model Score: `{score}`")

else:
    st.info("Please upload a CSV file from the sidebar to get started.")
