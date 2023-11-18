import streamlit as st
import pickle
import pandas as pd
import numpy as np
from utils import load_obj
from exception import CustomException
import os
import sys


# pip = pickle.load(open('pip.pkl_1','rb'))
model_path = os.path.join("artifacts", 'model.pkl')
preprocessor_path = os.path.join("artifacts", 'preprocess.pkl')

preprocessor = load_obj(preprocessor_path)
model = load_obj(model_path)



teams = [
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'
]

cities = ['Newlands',
 'Beausejour',
 'Kensington',
 'Shere',
 'Trent',
 'Wankhede',
 'Eden',
 'Dubai',
 'Warner',
 'Old',
 'Kennington',
 'Pallekele',
 'The',
 'Vidarbha',
 'Zahur',
 'Adelaide',
 'New',
 "Queen's",
 'SuperSport',
 'R.Premadasa',
 'Westpac',
 'Sydney',
 'R',
 "Lord's",
 'M',
 'Sophia',
 'Kingsmead',
 'Punjab',
 'Seddon',
 'Gaddafi',
 'Sheikh',
 'Central',
 'Bay',
 'Melbourne']

st.title('Cricket score predictor')

col1, col2 = st.columns(2)
with col1 :
   batting_team =  st.selectbox('Select batting team', sorted(teams))

with col2  :
   bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    Current_score = st.number_input('Current  score')
with col4:
    Overs_done = st.number_input(' Overs done(works for over >5) ')
with col5:
    wickets = st.number_input('wickets  out')
last_five = st.number_input('Runs scored in last five overs')

if st.button('Predict score'):
    balls_left = 120 - (Overs_done*6)
    wickets_left = 10 - wickets
    crr = Current_score/Overs_done

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city],
                             'current_score': [Current_score],'crr': [crr], 'balls_left': [balls_left],'wickets_left': [wickets_left],
                             'last_five': [last_five]})
    

    prerpocess_df = preprocessor.transform(input_df)
    result = model.predict(prerpocess_df)
    st.header('Predicted Score: ' + str(int(result[0])))











