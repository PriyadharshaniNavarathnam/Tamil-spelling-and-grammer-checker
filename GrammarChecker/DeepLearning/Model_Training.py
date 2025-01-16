import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def train_model(model, encoder_input_data, decoder_input_data, decoder_output_data, batch_size=64, epochs=100):
    # Train-test split
    (encoder_input_train, encoder_input_val,
     decoder_input_train, decoder_input_val,
     decoder_output_train, decoder_output_val) = train_test_split(
        encoder_input_data, decoder_input_data, decoder_output_data, test_size=0.2, random_state=42)

    history = model.fit(
        [encoder_input_train, decoder_input_train],
        decoder_output_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([encoder_input_val, decoder_input_val], decoder_output_val)
    )

    return history


def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()