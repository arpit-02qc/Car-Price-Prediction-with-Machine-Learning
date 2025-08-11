#Car Price Prediction with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Load dataset
df = pd.read_csv("car price.csv") 
print(df.head())
print(df.info())
print(df.describe())

#Clean and Preprocess Data
df.drop(columns=[col for col in ["car_ID", "symboling"] if col in df.columns], inplace=True)
# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Check for missing values
print(df.isnull().sum())
df.dropna(inplace=True)

#feature scaling
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model Training and Evaluation
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

xgb = XGBRegressor()
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
print("XGBoost R2:", r2_score(y_test, xgb_pred))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))

# visualize Performance
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest Predictions")
plt.show()

# Save model using joblib or pickle
import joblib
joblib.dump(rf, "car_price_model.pkl")

# Use Streamlit to build a UI
# streamlit run app.py