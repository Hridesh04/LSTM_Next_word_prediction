## import all the important libraries for the project to deploy in streamlit
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the trained model 
from tensorflow.keras.modls import load_model
model = load_model("next_word_lstm.h5")
## load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Funtion to predict the next word in a sequence
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # Ensure the input sequence is of the correct length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


## Streamlit app
st.title("Next Word Prediction App")
st.write("Enter a sequence of words, and the app will predict the next word.")
input_text = st.text_input("Input Text:")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1 # +1 because the model was trained on sequences of length max_sequence_len-1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted next word: {next_word}")
