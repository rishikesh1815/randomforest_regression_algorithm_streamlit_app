import streamlit as st
import joblib
import numpy as np

# load trained model
model = joblib.load("rf_house_price_model.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter house details 👇")

# inputs
area = st.number_input("Area (sqft)", min_value=300, max_value=5000)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10)
age = st.number_input("House Age (years)", min_value=0, max_value=50)

if st.button("Predict Price"):
    input_data = np.array([[area, rooms, age]])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: ₹ {int(prediction[0]):,}")
