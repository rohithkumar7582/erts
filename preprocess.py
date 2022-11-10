import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import streamlit_google_oauth as oauth
from dotenv import load_dotenv


import keras
from  keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd


from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,TFBertModel
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from custom_weights import CustomModel

from sklearn.model_selection import train_test_split


df = pd.read_csv('Emotion-Recognition-using-Text-with-Emojis-and-Speech/text_dataset/train.csv')

train,test = train_test_split(df, random_state=42, test_size=0.2)
train.shape,test.shape
train,val = train_test_split(train, random_state=42, test_size=0.1)
train.shape,val.shape

load_dotenv()

label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


'''def sparse_weighted_loss(target, output, weights):
      return tf.multiply(tf.keras.backend.sparse_categorical_crossentropy(target, output), weights)
custom_obj = {}
custom_obj['sparse_weighted_loss'] = sparse_weighted_loss(target=1,output=5,weights=2376)'''


bilstm_model = load_model("Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bi-lstm.h5")
bert_model = load_model("Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bert_model_04-11-2022.h5")
speech_model = load_model("Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_model.h5")


X = train['text']
bilstm_tokenizer=Tokenizer(15212,lower=True,oov_token='UNK')
bilstm_tokenizer.fit_on_texts(X)

encoded_dict  = {'anger':0,'fear':1, 'happy':2, 'neutral':3, 'sad':4}
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def get_key(value):
    dictionary={'happy':0,'angry':1,'neutral':2,'sad':3,'fear':4}
    for key,val in dictionary.items():
          if (val==np.argmax(value)):
            return key


def bilstm_predict(sentence):
  print("Original text:",sentence)
  sentence_lst=[]
  sentence_lst.append(sentence)
  sentence_seq=bilstm_tokenizer.texts_to_sequences(sentence_lst)
  print("preprocessed text : ",sentence_seq)
  sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
  ans=get_key(bilstm_model.predict(sentence_padded))
  return ans


def bilstm_preprocess(raw_text):
    return bilstm_predict(raw_text)


def bert_preprocess(raw_text):
    x_val = bert_tokenizer(
    text=[raw_text],
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
    bert_emotion = bert_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    return bert_emotion

def speech_preprocess(raw_speech):
    pass