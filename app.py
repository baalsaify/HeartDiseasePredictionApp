'''
Check <https://www.youtube.com/playlist?list=PL9jefoqM2f-OMXLVc> for more information
'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import base64

st.title("Heart Disease Predictor")
tab1, tab2 = st.tabs(['Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("sex", ['Male', 'Female'])
    chest_pain = st.selectbox('Chest Pain Type', ['Atypical Agina', 'Non-Aginal', 'Asymptomatic', 'Typical Agina'])
    resting_bp = st.number_input('Resting Blood Pressure (mm hg)', min_value=0, max_value=300)
    cholesterol = st.number_input('Serum Cholesterol (mm/dl)', min_value=0)
    fasting_bs = st.selectbox('Fasting Blood Sugar', ['<= 120 mg/dl', '> 120 mg/dl'])
    resting_ecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    max_hr = st.number_input('Maximum Heart Rate Acheived', min_value=60, max_value=200)
    excercise_agina = st.selectbox("Exercise-induced Agina", ['Yes', 'No'])
    oldpeak = st.number_input('Oldpeak (ST Depression)', min_value=0.0, max_value=10.0)
    st_slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloaping', 'Flat', 'Downsloaping'])

    sex = 0 if sex=='male' else 1
    chest_pain = ['Atypical Agina', 'Non-Aginal', 'Asymptomatic', 'Typical Agina'].index(chest_pain)
    fasting_bs = 1 if fasting_bs == '> 120 mg/dl' else 0
    resting_ecg = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(resting_ecg)
    excercise_agina = 1 if excercise_agina == 'Yes' else 0
    st_slope = ['Upsloaping', 'Flat', 'Downsloaping'].index(st_slope)

    data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [excercise_agina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algNames = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelNames = ['DT.pkl', 'LR.pkl', 'RF.pkl', 'svm.pkl']

    def DTPredict(data):
        model = pickle.load(open('DT.pkl', 'rb'))
        return model.predict(data)

    def LRPredict(data):
        model = pickle.load(open('LR.pkl', 'rb'))
        return model.predict(data)

    def RFPredict(data):
        model = pickle.load(open('RF.pkl', 'rb'))
        return model.predict(data)

    def SVMPredict(data):
        model = pickle.load(open('svm.pkl', 'rb'))
        return model.predict(data)

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
    with col1:
        if st.button('Logistic Regression'):
            result = LRPredict(data)
            st.markdown('----------------------')
            if result == 0:
                st.write('No heart disease detected')
            else:
                st.write('heart disease detected')
            st.markdown('----------------------')

    with col2:
        if st.button('Decision Tree'):
            result = DTPredict(data)
            st.markdown('----------------------')
            if result == 0:
                st.write('No heart disease detected')
            else:
                st.write('heart disease detected')
            st.markdown('----------------------')

    with col3:
        if st.button('Random Forest'):
            result = RFPredict(data)
            st.markdown('----------------------')
            if result == 0:
                st.write('No heart disease detected')
            else:
                st.write('heart disease detected')
            st.markdown('----------------------')

    with col4:
        if st.button('Support Vector Machine'):
            result = SVMPredict(data)
            st.markdown('----------------------')
            if result == 0:
                st.write('No heart disease detected')
            else:
                st.write('heart disease detected')
            st.markdown('----------------------')

with tab2:
    data = {'Logistic Regression': 85.86, 'SVM': 84.22, 'Decision Tree': 80.97, 'Random Forest': 87.50}
    models = list(data.keys())
    accuracies = list(data.values())
    df = pd.DataFrame({
        'Models': models,
        'Accuracies': accuracies
    })
    st.write('### Each model acheived accuracy')
    st.write(df)
    #fig = px.bar(df, x='Models', y='Accuracies')
    #st.plotly_chart(fig, use_container_width=True)
