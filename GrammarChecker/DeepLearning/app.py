import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Function to safely load models and tokenizers with error handling
def load_assets():
    try:
        encoder_model = tf.keras.models.load_model('models/encoder_model.h5', custom_objects=None)
        decoder_model = tf.keras.models.load_model('models/decoder_model.h5', custom_objects=None)
        
        with open('models/input_tokenizer.pkl', 'rb') as f:
            input_tokenizer = pickle.load(f)
        with open('models/output_tokenizer.pkl', 'rb') as f:
            output_tokenizer = pickle.load(f)

        return encoder_model, decoder_model, input_tokenizer, output_tokenizer
    except Exception as e:
        st.error(f"Failed to load models or tokenizers: {e}")
        return None, None, None, None

encoder_model, decoder_model, input_tokenizer, output_tokenizer = load_assets()

# Define constants based on your actual model training setup
MAX_INPUT_LENGTH = 100  # Adjust this based on your training setup
MAX_TARGET_LENGTH = 100  # Adjust this based on your training setup

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index.get('<start>', 1)
    decoded_sentence = ''
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '<unk>')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > MAX_TARGET_LENGTH:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

    return decoded_sentence.strip()

# Streamlit UI setup
st.title("Tamil Grammar Correction Tool")
st.write("Enter an ungrammatical Tamil sentence to see the corrected version:")

input_sentence = st.text_area("Enter Sentence Here", height=150)
if st.button('Correct Grammar'):
    if input_sentence:
        input_seq = pad_sequences(input_tokenizer.texts_to_sequences([input_sentence]), maxlen=MAX_INPUT_LENGTH, padding='post')
        predicted_sentence = decode_sequence(input_seq)
        st.write("Corrected Sentence:", predicted_sentence)
    else:
        st.warning('Please enter some text to correct.')

