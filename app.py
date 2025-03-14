import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load trained model
with open("parkinson_model.pkl", "rb") as f:
    scaler, pca, model = pickle.load(f)

# Streamlit UI
st.title("Parkinson’s Disease Prediction")
st.write("Provide patient data to predict Parkinson’s disease.")

# User input form
def user_input():
    input_data = {}
    feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                     'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 
                     'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 
                     'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

    for feature in feature_names:
        input_data[feature] = st.sidebar.slider(feature, float(-2), float(2), float(0))

    return pd.DataFrame([input_data])

# Get user input
user_df = user_input()

# Process input
if st.button("Predict"):
    input_scaled = scaler.transform(user_df)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)
    
    st.write("## Prediction:")
    st.write("🟢 No Parkinson's Detected" if prediction[0] == 0 else "🔴 Parkinson's Detected")
