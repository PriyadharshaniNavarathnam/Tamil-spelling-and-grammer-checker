import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.optimizers import Adam

# Load dataset
file_path = r'D:\7th_Semester_FoE_UoJ\EC9640_Artificial Intelligence\Project\GrammarChecker\DeepLearning\DataSet\TamilDatasetGrammar.xlsx'
df = pd.read_excel(file_path)

# Clean dataset
df = df.dropna(subset=['Original Sentence', 'Corrected Sentence'])
df['Original Sentence'] = df['Original Sentence'].astype(str).str.strip()
df['Corrected Sentence'] = df['Corrected Sentence'].astype(str).str.strip()

# Add start and end tokens for decoding
input_sentences = df['Original Sentence'].values
target_sentences = ['<start> ' + sentence + ' <end>' for sentence in df['Corrected Sentence'].values]

# Tokenize input and target sentences
input_tokenizer = Tokenizer(filters='', oov_token='<unk>')
input_tokenizer.fit_on_texts(input_sentences)
input_sequences = input_tokenizer.texts_to_sequences(input_sentences)

output_tokenizer = Tokenizer(filters='', oov_token='<unk>')
output_tokenizer.fit_on_texts(target_sentences)
target_sequences = output_tokenizer.texts_to_sequences(target_sentences)

# Vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

# Maximum sequence lengths
max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = max(len(seq) for seq in target_sequences)

# Pad sequences
encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

# Decoder output data
decoder_output_data = np.zeros((len(target_sequences), max_target_length, output_vocab_size), dtype='float32')
for i, seq in enumerate(target_sequences):
    for t, word_id in enumerate(seq):
        if t > 0:  # Skip the first token
            decoder_output_data[i, t - 1, word_id] = 1.0

# Train-test split
encoder_input_train, encoder_input_val, decoder_input_train, decoder_input_val, decoder_output_train, decoder_output_val = train_test_split(
    encoder_input_data, decoder_input_data, decoder_output_data, test_size=0.2, random_state=42)

# Model parameters
embedding_dim = 512  # Increased embedding dimension
hidden_units = 1024  # Increased hidden units
learning_rate = 0.001  # Adjusted learning rate

# Encoder
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(hidden_units, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_target_length,))
decoder_embedding = Embedding(output_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # Optimized Adam
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 100  # Increased epochs for better learning

history = model.fit(
    [encoder_input_train, decoder_input_train],
    decoder_output_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([encoder_input_val, decoder_input_val], decoder_output_val)
)

# Encoder model for inference
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model for inference
decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embedding_inf = Embedding(output_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Decode function
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '<unk>')
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_target_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

# New function to check a paragraph
def check_paragraph(paragraph):
    sentences = re.split(r'[.!?]', paragraph)  # Split the paragraph into sentences
    sentences = [sentence.strip() for sentence in sentences if sentence]  # Clean up empty sentences
    corrected_sentences = []

    for sentence in sentences:
        input_seq = pad_sequences(input_tokenizer.texts_to_sequences([sentence]), maxlen=max_input_length, padding='post')
        corrected_sentence = decode_sequence(input_seq)
        corrected_sentences.append(corrected_sentence)

    corrected_paragraph = ' '.join(corrected_sentences)
    print(f"\nOriginal Paragraph:\n{paragraph}")
    print(f"\nCorrected Paragraph:\n{corrected_paragraph}")

# Example usage
input_paragraph = input("சமைத்தார் தந்தை சாப்பாட்டை சமையலறையில். கடிதத்தை எழுதியது அவன் நண்பனுக்கு. பாடினாள் குயில் மாலை பொழுதில் பாடலை. அழைத்தார் சாலையில் நண்பனை கூட்டத்தில் அவள். பரிசு கொடுத்தாள் தந்தைக்கு மாணவர்கள் வீட்டில் அவர். விளையாடினாள் தோட்டத்தில் சிறுவர்கள். கூறினான் பாட்டி மாலை நேரத்தில் கதையை. பார்த்தாள் தோட்டத்தில் அவளை சிறுவர்கள். தூக்கியான் அவள் கையை. வாசித்தாள் புத்தகத்தை அவன். சந்தித்தாள் அவள் தோழனை.\n")
check_paragraph(input_paragraph)
