import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons.data")
    return df
df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    with open("parkinson_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the saved model
model = load_model()

# Feature scaling and PCA
X = df.drop(["status", "name"], axis=1)
y = df["status"]

scaler = MinMaxScaler((-1, 1))
X_scaled = scaler.fit_transform(X)

pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

# Streamlit UI
st.title("Parkinson's Disease Prediction")
st.write("This app predicts Parkinson's Disease based on patient voice measures.")
st.write("Attribute Information : Matrix column entries (attributes): ***name*** - ASCII subject name and recording number ***MDVP:Fo(Hz)***- Average vocal fundamental frequency ***MDVP:Fhi(Hz)*** - Maximum vocal fundamental frequency **MDVP:Flo(Hz)** - Minimum vocal fundamental frequency ***MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP*** - Several measures of variation in fundamental frequency ***MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA*** - Several measures of variation in amplitude ***NHR,HNR***- Two measures of ratio of noise to tonal components in the voice **status** - Health status of the subject (one) - Parkinson's, (zero) - healthy (Dependent variable.) ***RPDE,D2*** - Two nonlinear dynamical complexity measures ***DFA*** - Signal fractal scaling exponent ***spread1,spread2,PPE*** - Three nonlinear measures of fundamental frequency variation")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Sidebar Inputs
st.sidebar.header("Input Features")
input_features = np.array([st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in df.columns[1:-1]])

# Prediction
if st.button("Predict"):
    input_scaled = scaler.transform([input_features])
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)
    
    st.write("### Prediction:")
    st.write("🟢 No Parkinson's Detected" if prediction[0] == 0 else "🔴 Parkinson's Detected")
