import streamlit as st
import numpy as np
import joblib

# Load the saved model, scaler, and label encoder
model = joblib.load('traffic_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Set page configuration
st.set_page_config(
    page_title="TRAFFIC FLOW PREDICTION",
    page_icon="🚦",
    layout="centered",
)

# Add custom CSS for consistent fonts
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        background-color: #f7f9fc;
    }
    .header-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a4a4a;
        text-align: center;
    }
    .description-text, .sidebar-text, .result-text, .footer-text {
        font-size: 1.2rem;
        color: #6a6a6a;
    }
    .result-text {
        font-size: 1.8rem;
        color: #2a9d8f;
        font-weight: bold;
    }
    .sidebar-text {
        font-size: 1rem;
    }
    .footer-text {
        font-size: 0.9rem;
        color: #aaaaaa;
        text-align: center;
    }
    .sidebar .block-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 1rem;
        color: #4a4a4a;
    }
    </style>
""", unsafe_allow_html=True)

# Add a header and description
st.markdown('<div class="header-text">🚦 TRAFFIC FLOW PREDICTION</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="description-text">
   <BR>  
    Provide details about the vehicle counts, and the app will predict the traffic situation: 
    <i>low, normal, or heavy</i>.  
    </div>
""", unsafe_allow_html=True)

# Create a sidebar
st.sidebar.markdown('<div class="sidebar-text"><b>VEHICLE COUNTS:</b></div>', unsafe_allow_html=True)
st.sidebar.write('<div class="sidebar-text">Adjust the values for the vehicle counts:</div>', unsafe_allow_html=True)

# Input fields for features (in the sidebar)
car_count = st.sidebar.number_input("🚗 Car Count:", min_value=0, value=50)
bike_count = st.sidebar.number_input("🏍️ Bike Count:", min_value=0, value=10)
bus_count = st.sidebar.number_input("🚌 Bus Count:", min_value=0, value=50)
truck_count = st.sidebar.number_input("🚛 Truck Count:", min_value=0, value=20)

# Add a button to trigger prediction
if st.sidebar.button("🔍 Predict Traffic Situation"):
    # Prepare the input
    test_input = np.array([[car_count, bike_count, bus_count, truck_count]])
    test_input_scaled = scaler.transform(test_input)
    predicted_class = model.predict(test_input_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    # Display the result
    st.markdown('<div class="result-text">🚦 Prediction Result</div>', unsafe_allow_html=True)
    st.success(f"The predicted traffic situation is: **{predicted_label[0].capitalize()}**")

