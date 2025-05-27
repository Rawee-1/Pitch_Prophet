import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('IPL Win Predictor')
st.image("image.jpg")


pipe = pickle.load(open('pipe32111.pkl', 'rb'))

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

batting_team = st.selectbox('Select the batting team', sorted(teams))
bowling_team = st.selectbox('Select the bowling team', sorted(teams))
selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target score')
score = st.number_input('Current score')
overs = st.number_input('Overs completed')
wickets_out = st.number_input('Wickets fallen')


if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets_out
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.subheader(f"{batting_team} - {round(win * 100)}% chance of winning")
    st.subheader(f"{bowling_team} - {round(loss * 100)}% chance of winning")









