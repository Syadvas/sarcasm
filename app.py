import json
from flask import Flask, request
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)


def model_predict(sentence):
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return model.predict(padded)

model = tf.keras.models.load_model('srcsm_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    _tokenizer = pickle.load(handle)

tokenizer.word_index = _tokenizer


app = Flask(__name__)

@app.route("/result")
def result():
    sentence = ["this is good!","second sentence"]
    x = model_predict(sentence)
    
    return str(x)

if __name__ =='__main__':
    app.run()