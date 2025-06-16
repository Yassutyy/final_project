import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🚗 Car Price Prediction App")
st.markdown("Predict used car prices based on brand, fuel type, year, and kilometers driven.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("car_data.csv")
    df.columns = ['Brand', 'Year', 'Selling_Price', 'KM_Driven', 'Fuel']
    return df

df = load_data()

# Encode categorical features
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])

# Features and target
X = df[['Brand', 'Year', 'KM_Driven', 'Fuel']]
y = df['Selling_Price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# User input
st.header("Enter Car Details")
brand = st.selectbox("Select Car Brand", le_brand.classes_)
fuel = st.selectbox("Select Fuel Type", le_fuel.classes_)
year = st.number_input("Enter Year of Manufacture", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Enter Kilometers Driven", min_value=0, value=50000)

# Prediction
if st.button("Predict Price"):
    brand_encoded = le_brand.transform([brand])[0]
    fuel_encoded = le_fuel.transform([fuel])[0]
    input_data = np.array([[brand_encoded, year, km_driven, fuel_encoded]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹{int(prediction):,}")
