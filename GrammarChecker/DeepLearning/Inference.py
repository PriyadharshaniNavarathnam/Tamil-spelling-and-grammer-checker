import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from nltk.translate.bleu_score import sentence_bleu


def build_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense, hidden_units):
    # Encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder inference model
    decoder_state_input_h = Input(shape=(hidden_units,))
    decoder_state_input_c = Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding_inf = decoder_embedding(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, output_tokenizer, max_target_length):
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