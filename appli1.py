import streamlit as st
import numpy as np
import pickle

# Load the trained model & encoders
with open("car_price_model.pkl", "rb") as f:
    model, scaler, label_encoders = pickle.load(f)

# Streamlit UI
st.title("🚗 Car Price Prediction App")

st.sidebar.header("Enter Car Details")
year = st.sidebar.number_input("Year", min_value=2000, max_value=2025, value=2019)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50001)
engine_cc = st.sidebar.number_input("Engine Size (CC)", min_value=500, max_value=6000, value=1504)

fuel = st.sidebar.selectbox("Fuel Type", label_encoders["fuel"].classes_)
seller_type = st.sidebar.selectbox("Seller Type", label_encoders["seller_type"].classes_)
transmission = st.sidebar.selectbox("Transmission", label_encoders["transmission"].classes_)
owner = st.sidebar.selectbox("Owner Type", label_encoders["owner"].classes_)

# Encode categorical inputs
def safe_encode(column, value):
    if value not in label_encoders[column].classes_:
        label_encoders[column].classes_ = np.append(label_encoders[column].classes_, value)
    return label_encoders[column].transform([value])[0]

fuel_encoded = safe_encode('fuel', fuel)
seller_encoded = safe_encode('seller_type', seller_type)
transmission_encoded = safe_encode('transmission', transmission)
owner_encoded = safe_encode('owner', owner)

# Prepare input data
input_data = np.array([[year, km_driven, fuel_encoded, seller_encoded, transmission_encoded, owner_encoded, engine_cc]])
input_data[:, [0, 1, 6]] = scaler.transform(input_data[:, [0, 1, 6]])

# Predict
if st.sidebar.button("Predict Car Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ${round(predicted_price, 2)}")
