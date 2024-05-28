import sys
sys.path = sorted(sys.path, key=lambda x: 'site-packages' in x)
import streamlit as st
import pandas as pd
import pickle


st.markdown("""

    <h1 style='text-align: center;'>Diabetes Prediction App</h1>
    <h2 style='text-align: center;'>M.Guenbour - M.Anchaoui</h2>
    <h4 style='text-align: center;'>MSDE4 : Cloud Computing Project</h4>
<h3 style='text-align: center;'>*** *** ***</h3>
  
""", unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 10)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 168)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 130, 74)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 0)
    Insulin = st.sidebar.slider('Insulin', 0, 900, 0)
    BMI = st.sidebar.slider('BMI', 0.0, 70.0, 38.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.000, 2.500,0.537)
    Age = st.sidebar.slider('Age', 20, 90, 34)
    
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parameters')
st.dataframe(df, hide_index=True)

# Load the model
model_diabetes = pickle.load(open('model.pkl', 'rb'))


# Predict
prediction = model_diabetes.predict(df)
prediction_proba = model_diabetes.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)
st.markdown("""
   <p style='text-align: center; font-size: 12px;'>0 : non-diabetic &emsp;1: Diabetic</p>
""", unsafe_allow_html=True)

st.subheader('Prediction Probability')
st.write(prediction_proba)
