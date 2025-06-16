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
df = pd.read_csv("car_data_set.csv")

st.set_page_config(layout="wide")

# Sidebar layout
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📁 Dataset", "📊 Visualizations", "🧠 Predictor"])

# App title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🚗 Car Price Prediction Tool</h1>", unsafe_allow_html=True)

# Home Page
if option == "🏠 Home":
    st.markdown("""
    ### 🔧 About This Tool
    This application helps you:
    - Explore the dataset used to train a car price prediction model.
    - Visualize key insights using interactive graphs.
    - Predict the **selling price** of a car using:
        - Linear Regression
        - Random Forest Regression
    
    Use the sidebar to navigate ➡️
    """)

# Dataset Viewer
elif option == "📁 Dataset":
    st.subheader("🔍 Training Dataset")
    st.dataframe(df)

# Visualizations
elif option == "📊 Visualizations":
    st.subheader("📊 Data Visualizations")

    fig1 = px.histogram(df, x='Selling_Price', nbins=50, title='Selling Price Distribution', marginal="box")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x='Fuel', y='Selling_Price', title='Selling Price by Fuel Type')
    st.plotly_chart(fig2, use_container_width=True)

# Predictor
elif option == "🧠 Predictor":
    st.subheader("⚙️ Choose Prediction Model")
    model_choice = st.radio("Select Model", ["Linear Regression", "Random Forest"])

    st.markdown("### 📝 Input Car Details")
    brand = st.selectbox("Select Car Brand", df['Brand'].unique())
    year = st.slider("Manufacturing Year", 1995, 2025, 2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
    fuel = st.selectbox("Select Fuel Type", df['Fuel'].unique())

    if st.button("🚀 Predict Selling Price"):
        try:
            brand_encoded = brand_encoder.transform([brand])[0]
            fuel_encoded = fuel_encoder.transform([fuel])[0]
            car_age = 2025 - year
            features = [[brand_encoded, car_age, km_driven, fuel_encoded]]

            if model_choice == "Linear Regression":
                prediction = model_lr.predict(features)[0]
                st.success(f"💸 Predicted Price (Linear Regression): ₹ {int(prediction):,}")
            else:
                prediction = model_rf.predict(features)[0]
                st.success(f"💸 Predicted Price (Random Forest): ₹ {int(prediction):,}")
        except Exception as e:
            st.error("Prediction failed. Please check your input or model files.")
