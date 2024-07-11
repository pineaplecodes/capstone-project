import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_data():
    df = pd.read_csv('hungarian_cleaned.csv',
                     delimiter=',', on_bad_lines='skip')
    return df


@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler


# Title
st.title("Heart Disease Prediction")

# Sidebar for user input
st.sidebar.header('User Input Parameters')


def user_input_features():
    age = st.sidebar.slider('Age', 28, 66, 29)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest pain type', ('Typical angina',
                              'Atypical angina', 'Non-anginal pain', 'Asymptomatic'))
    trestbps = st.sidebar.slider(
        'Resting blood pressure (in mm Hg on admission to the hospital)', 92, 200, 120)
    chol = st.sidebar.slider('Serum cholestoral in mg/dl', 126, 564, 240)
    fbs = st.sidebar.selectbox(
        'Fasting blood sugar > 120 mg/dl', ('True', 'False'))
    restecg = st.sidebar.selectbox('Resting electrocardiographic results', (
        'Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'))
    thalach = st.sidebar.slider('Maximum heart rate achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise induced angina', ('Yes', 'No'))
    oldpeak = st.sidebar.slider(
        'ST depression induced by exercise relative to rest', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox(
        'Slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.slider(
        'Number of major vessels (0-3) colored by fluoroscopy', 0, 3, 0)
    thal = st.sidebar.selectbox(
        'Thalassemia', ('Normal', 'Fixed defect', 'Reversable defect'))

    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == 'True' else 0,
        'restecg': restecg,
        'thalach': thalach,
        'exang': 1 if exang == 'Yes' else 0,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()


input_df['cp'] = input_df['cp'].map(
    {'Typical angina': 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3})
input_df['restecg'] = input_df['restecg'].map(
    {'Normal': 0, 'Having ST-T wave abnormality': 1, 'Showing probable or definite left ventricular hypertrophy': 2})
input_df['slope'] = input_df['slope'].map(
    {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2})
input_df['thal'] = input_df['thal'].map(
    {'Normal': 0, 'Fixed defect': 1, 'Reversable defect': 2})


model, scaler = load_model()


input_df_scaled = scaler.transform(input_df)

st.subheader('User Input as DataFrame')
st.write(input_df)


if st.button('Predict'):

    prediction = model.predict(input_df_scaled)

    st.subheader('Prediction:')
    st.write(
        'Positive for heart disease' if prediction[0] == 1 else 'Negative for heart disease')
