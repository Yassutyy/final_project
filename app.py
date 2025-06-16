import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load encoders and models
with open("brand_encoder.pkl", "rb") as f:
    brand_encoder = pickle.load(f)
with open("fuel_encoder.pkl", "rb") as f:
    fuel_encoder = pickle.load(f)
with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)
with open("model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Load dataset
df = pd.read_csv("car_data.csv")

st.set_page_config(layout="wide")

# Sidebar layout
st.sidebar.title("Car Price Predictor")
option = st.sidebar.radio("Navigation", ["Dataset", "Visualizations", "Predictor"])

st.title("🚗 Car Price Prediction Tool")

if option == "Dataset":
    st.subheader("🔍 Training Dataset")
    st.dataframe(df)

elif option == "Visualizations":
    st.subheader("📊 Data Visualizations")

    # Selling Price Distribution
    fig1 = px.histogram(df, x='Selling_Price', nbins=50, title='Selling Price Distribution', marginal="box")
    st.plotly_chart(fig1, use_container_width=True)

    # Selling Price by Fuel Type
    fig2 = px.box(df, x='Fuel', y='Selling_Price', title='Selling Price by Fuel Type')
    st.plotly_chart(fig2, use_container_width=True)

elif option == "Predictor":
    st.subheader("⚙️ Model Selection")
    model_choice = st.radio("Choose Model", ["Linear Regression", "Random Forest"])

    brand = st.selectbox("Select Car Brand", df['Brand'].unique())
    year = st.slider("Manufacturing Year", 1995, 2025, 2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
    fuel = st.selectbox("Select Fuel Type", df['Fuel'].unique())

    if st.button("Predict Selling Price"):
        brand_encoded = brand_encoder.transform([brand])[0]
        fuel_encoded = fuel_encoder.transform([fuel])[0]
        car_age = 2025 - year

        features = [[brand_encoded, car_age, km_driven, fuel_encoded]]

        if model_choice == "Linear Regression":
            prediction = model_lr.predict(features)[0]
            st.success(f"💸 Predicted Selling Price (Linear Regression): ₹ {int(prediction):,}")

        elif model_choice == "Random Forest":
            prediction = model_rf.predict(features)[0]
            st.success(f"💸 Predicted Selling Price (Random Forest): ₹ {int(prediction):,}")
