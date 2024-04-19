import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from predictions import ordinal_encoding, get_prediction


# Load the model
model = joblib.load('./Notebook/ran_for_new.joblib')

st.set_page_config(page_title="Income Limit Prediction App",
                   page_icon="🚧", layout="wide")

#creating drop-down menu

options_education=[' High school graduate', ' 12th grade no diploma', ' Children',
       ' Bachelors degree(BA AB BS)', ' 7th and 8th grade', ' 11th grade',
       ' 9th grade', ' Masters degree(MA MS MEng MEd MSW MBA)',
       ' 10th grade', ' Associates degree-academic program',
       ' 1st 2nd 3rd or 4th grade', ' Some college but no degree',
       ' Less than 1st grade', ' Associates degree-occup /vocational',
       ' Prof school degree (MD DDS DVM LLB JD)', ' 5th or 6th grade',
       ' Doctorate degree(PhD EdD)']

options_gender=['Male', 'Female']


st.markdown("<h1 style='text-align: center;'>Income Limit Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")

        occupation_code=st.slider("occupation code: ", 0, 46, value=0, format="%d")
        age=st.slider("age: ", 0, 90, value=0, format="%d")
        working_week_per_year=st.slider("working week per year: ", 0, 52, value=0, format="%d")
        industry_code=st.slider("industry code: ", 0, 51, value=0, format="%d")
        total_employed=st.slider("total employed: ", 0, 6, value=0, format="%d")
        importance_of_record=st.slider("importance of record: ", 37, 18656, value=0, format="%d")
        stock_status=st.slider("stock status: ", 0, 99999, value=0, format="%d")
        education=st.selectbox("education: ", options_education)
        gender=st.selectbox("gender: ", options_gender)
        gains=st.slider("gains: ", 0, 99999, value=0, format="%d")

        submit = st.form_submit_button("Predict")

    if submit:
        #ordinal encoding
        ordinal_education=ordinal_encoding(education, options_education)
        ordinal_gender=ordinal_encoding(gender, options_gender)

        data=np.array([occupation_code, age, working_week_per_year, 
                       industry_code, total_employed, importance_of_record, stock_status, ordinal_education, ordinal_gender, gains]).reshape(1, -1)
        
        pred=get_prediction(data=data, model=model)

        st.write(f"The predicted income level is: {pred[0]}")

if __name__ == '__main__':
    main()