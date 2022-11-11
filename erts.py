import streamlit as st
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import os
import streamlit_google_oauth as oauth
from dotenv import load_dotenv
from preprocess import *
import time
import pymongo
from record import audiorec_demo_app

load_dotenv()

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils




#bilstm_model = load_model("Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bi-lstm.h5")
#bert_model = "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bert_model_04-11-2022.h5"
#speech_model = "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_model.h5"

username = ""
login_start_time = time.time()

login_info = oauth.login(
        client_id=os.environ['CLIENT_ID'],
        client_secret=os.environ['CLIENT_SECRET_KEY'],
        redirect_uri=os.environ['CLIENT_REDIRECT_URI'],
        login_button_text="Continue with Google",
        logout_button_text="Logout",
    )

if login_info:
	login_start_time = time.time()
	user_id, user_email = login_info
	st.write(f"Welcome {user_email}")
	username = user_email

else:
	st.write("Please login to continue")


st.header("Emotion Recognizer using Text & Speech")

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

text_data = st.text_input("Enter Text")
text_predict_btn = st.button("Predict Emotion")

audio_record_btn = st.button("Record Audio")

if text_data and st.session_state["run"] != "false" and text_predict_btn:
	#webrtc_streamer(key="key", desired_playing_state=True,
	#			video_processor_factory=EmotionProcessor)
	st.write("Predicting ...")

if text_predict_btn:
	if not(emotion) and not(text_data):
		st.warning("Please enter a text or record audio to predict emotion !!")
		st.session_state["run"] = "true"
	else:

		bilstm_result = bilstm_preprocess(text_data)
		st.write("BILSTM emotion : ",bilstm_result)

		bert_result = bert_preprocess(text_data)
		st.write("BERT emotion : ",bert_result)

		webbrowser.open(f"https://www.youtube.com/results?search_query={bilstm_result}+song")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"

if audio_record_btn:
	# DESIGN implement changes to the standard streamlit UI/UX
	# Design move app further up and remove top padding
	st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
		unsafe_allow_html=True)
	# Design change st.Audio to fixed height of 45 pixels
	st.markdown('''<style>.stAudio {height: 45px;}</style>''',
		unsafe_allow_html=True)
	# Design change hyperlink href link color
	st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
		unsafe_allow_html=True)  # darkmode
	st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
		unsafe_allow_html=True)  # lightmode

	audiorec_demo_app()