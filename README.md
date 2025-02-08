# DSP-01-Final-Project
# Energy Consumption and Traffic Density Prediction
This project is a Streamlit-based web application that predicts Energy Consumption and Traffic Density based on various input features such as vehicle type, weather, economic condition, and more. The model uses Random Forest Regression to make predictions, with preprocessing steps to handle numerical and categorical features.

Features
Energy Consumption Prediction: Predicts energy consumption based on factors like vehicle type, speed, weather, and more.
Traffic Density Prediction: Predicts traffic density using similar features, helping to understand congestion patterns.
Requirements
To run this project, make sure you have the following libraries installed:

streamlit
pandas
numpy
scikit-learn

Dataset
The application uses the following columns from the dataset:

City
Vehicle Type
Weather
Economic Condition
Day Of Week
Hour Of Day
Speed
Is Peak Hour
Random Event Occurred
Energy Consumption (target variable)
Traffic Density (target variable)

Future Work
Improve model performance by exploring other regression models (e.g., Gradient Boosting, XGBoost).
Include more feature engineering for better prediction accuracy.
Visualize the results for better insights.
