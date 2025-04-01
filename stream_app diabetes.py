import streamlit as st
import pickle
import numpy as np

# Load the trained KNeighborsClassifier model
with open('diabetes_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the web app
st.title("Diabetes Prediction using KNeighborsClassifier")

# Input fields for the features using sliders
pregnancies = st.slider('Pregnancies', min_value=0, max_value=10, value=0)
glucose = st.slider('Glucose', min_value=0, max_value=200, value=0)
blood_pressure = st.slider('Blood Pressure', min_value=90, max_value=120, value=0)
skin_thickness = st.slider('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.slider('Insulin', min_value=0, max_value=900, value=0)
bmi = st.slider('BMI', min_value=0.0, max_value=70.0, value=0.0)
dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.slider('Age', min_value=0, max_value=120, value=0)

# Predict button
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    
    #result
    if prediction[0] == 1:
        st.write("The model predicts that this person is likely to have diabetes.")
    else:
        st.write("The model predicts that this person is unlikely to have diabetes.")

