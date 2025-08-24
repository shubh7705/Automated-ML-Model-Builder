import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

def train_and_display(model, model_name, X_train, X_test, y_train, y_test, is_classification=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if is_classification:
        score = accuracy_score(y_test, y_pred)
        metric = "Accuracy Score"
    else:
        score = r2_score(y_test, y_pred)
        metric = "R² Score"

    # Serialize the model
    pkl_data = pickle.dumps(model)

    # Download button
    st.download_button(
        label=f"Download {model_name}",
        data=pkl_data,
        file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
        mime="application/octet-stream"
    )

    # Display model score
    st.subheader(f"📈 {model_name} Performance")
    st.write(f"{metric}: `{score}`")


def model_fitting(X_train, X_test, y_train, y_test, dt):
    if dt == 'O':
        # Classification models
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            model = LogisticRegression()
            train_and_display(model, "Logistic Regression", X_train, X_test, y_train, y_test, is_classification=True)

        with col2:
            model = KNeighborsClassifier(n_neighbors=3)
            train_and_display(model, "K-Nearest Neighbors (KNN)", X_train, X_test, y_train, y_test, is_classification=True)

        with col3:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            train_and_display(model, "Random Forest Classifier", X_train, X_test, y_train, y_test, is_classification=True)

        with col4:
            model = DecisionTreeClassifier(random_state=1)
            train_and_display(model, "Decision Tree Classifier", X_train, X_test, y_train, y_test, is_classification=True)

    else:
        # Regression models
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            model = LinearRegression()
            train_and_display(model, "Linear Regression", X_train, X_test, y_train, y_test, is_classification=False)

        with col2:
            model = DecisionTreeRegressor(max_depth=2)
            train_and_display(model, "Decision Tree Regressor", X_train, X_test, y_train, y_test, is_classification=False)

        with col3:
            model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
            train_and_display(model, "Random Forest Regressor", X_train, X_test, y_train, y_test, is_classification=False)

        with col4:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            train_and_display(model, "ElasticNet Regressor", X_train, X_test, y_train, y_test, is_classification=False)


def na_values(data):
    return data.isna().sum().sum()

@st.cache_data
def split(X,y):
        if X.empty or y.empty:
            raise ValueError("X or y is empty. Cannot split empty dataset.")
        return train_test_split(X, y, test_size=0.2)

