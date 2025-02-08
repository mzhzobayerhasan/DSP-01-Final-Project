import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("your_data.csv")  # Replace with actual file path
        return data.dropna()  # Remove rows with missing values
    except FileNotFoundError:
        st.error("Error: Dataset file not found!")
        return None

# Train model function
@st.cache_resource
def train_model(data):
    if data is None:
        return None, None, [], []

    target_columns = ['Energy Consumption', 'Traffic Density']
    
    num_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove target columns if they exist
    available_targets = [col for col in target_columns if col in data.columns]
    if not available_targets:
        st.error("Target columns not found in dataset!")
        st.stop()

    num_features = [col for col in num_features if col not in available_targets]
    cat_features = [col for col in cat_features if col not in available_targets]

    # Preprocessing pipelines
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]) if cat_features else 'passthrough'

    preprocessing_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Split data
    X = data.drop(available_targets, axis=1)
    y = data[available_targets[0]]  # Only predicting 'Energy Consumption'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model pipeline
    model_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)

    # Predictions and RMSE
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model_pipeline, rmse, num_features, cat_features

# Load and train model
data = load_data()
if data is not None:
    model, rmse, num_features, cat_features = train_model(data)

    # Streamlit UI
    st.title("🔋 Energy Consumption Prediction App")
    st.write(f"📊 Model RMSE: **{rmse:.2f}**")

    # Sidebar inputs
    st.sidebar.header("Enter Feature Values")

    user_input = {}
    for feature in num_features:
        user_input[feature] = st.sidebar.number_input(f"{feature}", value=float(data[feature].mean()))

    for feature in cat_features:
        user_input[feature] = st.sidebar.selectbox(f"{feature}", data[feature].unique())

    # Predict button
    if st.sidebar.button("🔮 Predict"):
        user_df = pd.DataFrame([user_input])
        prediction = model.predict(user_df)
        st.sidebar.success(f"Predicted Energy Consumption: **{prediction[0]:.2f}**")
