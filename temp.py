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
import sounddevice as sd
from scipy.io.wavfile import write
from audio_recorder_streamlit import audio_recorder

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

text_data = st.text_input("Enter Text")
text_predict_btn = st.button("Predict Emotion")



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
		#np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"

#st.session_state["run"] = "true"

audio_record_btn = st.button("Record Audio")

if audio_record_btn:
	st.write("audio button clicked")
	st.write("Recording Started ...")
	print("Starting: Speak now!")
	audio_bytes = audio_recorder()


	st.audio(audio_bytes, format="audio/wav")
	st.session_state["run"] = "false"

	start = st.button("Start")
	stop = st.button("Stop")
	#reset = st.button("Reset")
	#download = st.button("Download")
	if start and audio_record_btn:

		fs = 44100  # this is the frequency sampling; also: 4999, 64000
		#myrecording = sd.rec(int(fs*10),samplerate=fs, channels=2)
		st.write("Recording Started ...")
		print("Starting: Speak now!")
		audio_bytes = audio_recorder()
		#sd.wait()

		if stop and audio_bytes:
			st.write("Recording finished")
			print("recording finished")
			st.audio(audio_bytes, format="audio/wav")
			#write(username+'_rec.wav', fs, myrecording)

			#st.audio(myrecording, format='audio/wav', start_time=0)



	

