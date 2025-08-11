# 🚗 Car Price Prediction with Machine Learning

This project predicts the **selling price of cars** based on various features such as year, fuel type, transmission, and kilometers driven.  
It uses **machine learning regression models** to learn from historical data and estimate prices for new cars.

## 📊 Dataset
- Features:
  - `Car_Name` – Model name
  - `Year` – Year of manufacture
  - `Selling_Price` – Price of the car (Target variable)
  - `Present_Price` – Current showroom price
  - `Kms_Driven` – Kilometers driven
  - `Fuel_Type` – Petrol/Diesel/CNG
  - `Seller_Type` – Dealer/Individual
  - `Transmission` – Manual/Automatic
  - `Owner` – Number of previous owners
- Rows: ~300 (depending on dataset used)

## 🛠 Steps Followed
1. Data loading & cleaning
2. Encoding categorical variables
3. Splitting into train & test sets
4. Training a **Random Forest Regressor**
5. Evaluating with R² Score, MAE, MSE, RMSE
6. Visualizing feature importance

## 📈 Results
- **R² Score:** ~0.95 (may vary)
- **MAE:** Low error margin
- **RMSE:** Minimal deviation from actual values

## 📦 Installation
```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
