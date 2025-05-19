import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Path to the dataset
DATASET_PATH = "/kaggle/input/recipe-dataset/preprocessed_recipe.csv"

# Model parameters
BATCH_SIZE = 32  # Reduced batch size
BUFFER_SIZE = 1000  # Smaller buffer
EMBEDDING_DIM = 128  # Reduced embedding size
RNN_UNITS = 256  # Reduced RNN units
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 50  # Fixed max sequence length
TEMPERATURE = 1.0
EOS_TOKEN = "<EOS>"
MAX_VOCAB_SIZE = 10000  # Limit vocabulary size


def load_data(max_recipes=80000):
    """Load and preprocess the dataset with memory constraints"""
    print("Loading dataset...")

    # Load the dataset in chunks to avoid memory issues
    df = pd.read_csv(DATASET_PATH, nrows=max_recipes)

    # Create the target format: "Ingredients: xxx | Recipe Steps: xxx"
    df["text_for_model"] = df.apply(
        lambda row: f"Ingredients: {row['ingredients_text']} | Recipe Steps: {row['steps_string_standardize']} {EOS_TOKEN}",
        axis=1,
    )

    # Get all the texts
    texts = df["text_for_model"].tolist()

    print(f"Dataset loaded with {len(texts)} recipes.")
    return texts


def create_tokenizer(texts):
    """Create and fit a tokenizer on the texts with limits"""
    print("Creating tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters="", oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    # Add EOS token to vocabulary if not there
    if EOS_TOKEN not in tokenizer.word_index:
        tokenizer.word_index[EOS_TOKEN] = len(tokenizer.word_index) + 1
        tokenizer.index_word[len(tokenizer.word_index)] = EOS_TOKEN

    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    print(f"Vocabulary size: {vocab_size}")
    return tokenizer, vocab_size


def batch_sequence_generator(texts, tokenizer, vocab_size, batch_size=1000):
    """Generate sequences in batches to prevent memory overflow"""
    print("Creating sequences in batches...")

    sequences = tokenizer.texts_to_sequences(texts)

    for batch_idx in range(0, len(sequences), batch_size):
        batch_sequences = sequences[batch_idx : batch_idx + batch_size]

        # Create input-target pairs for next word prediction
        input_sequences = []
        targets = []

        for seq in batch_sequences:
            # Limit sequence length to avoid memory issues
            if len(seq) > MAX_SEQUENCE_LENGTH + 1:
                seq = seq[: MAX_SEQUENCE_LENGTH + 1]

            for i in range(1, len(seq)):
                # Input is all words up to the target word
                n_gram_sequence = seq[:i]
                # Target is the next word
                target_word = seq[i]

                # Only use sequences that aren't too short
                if len(n_gram_sequence) >= 2:  # At least 2 words for input
                    input_sequences.append(n_gram_sequence)
                    targets.append(target_word)

        if not input_sequences:
            continue

        # Pad sequences (input only, not targets)
        padded_sequences = pad_sequences(
            input_sequences, maxlen=MAX_SEQUENCE_LENGTH - 1, padding="pre"
        )

        # Convert targets to one-hot encoding
        targets = np.array(targets)
        targets_one_hot = tf.keras.utils.to_categorical(targets, num_classes=vocab_size)

        print(f"Created batch of {len(padded_sequences)} sequences")
        yield padded_sequences, targets_one_hot


class RecipeRNNModel(tf.keras.Model):
    """Recipe generation RNN model (simplified)"""

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(
            rnn_units
        )  # No return_sequences for simple next word prediction
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.rnn(x)
        return self.dense(x)


def build_model(vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS):
    """Build the RNN model using the functional API for clarity"""
    print("Building model...")

    inputs = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH - 1,))
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    rnn_output = tf.keras.layers.SimpleRNN(rnn_units)(embedding)  # No return_sequences
    outputs = tf.keras.layers.Dense(vocab_size)(rnn_output)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def generate_text(model, tokenizer, seed_text, max_length=100, temperature=TEMPERATURE):
    """Generate recipe text based on ingredients"""
    print(f"Generating recipe from: '{seed_text}'")

    # Convert seed text to tokens
    input_tokens = tokenizer.texts_to_sequences([seed_text])[0]

    # Initialize text generation
    generated_text = seed_text

    for i in range(max_length):
        # Truncate input if too long
        if len(input_tokens) > MAX_SEQUENCE_LENGTH - 1:
            input_tokens = input_tokens[-(MAX_SEQUENCE_LENGTH - 1) :]

        # Pad the input to the required length
        padded_tokens = pad_sequences(
            [input_tokens], maxlen=MAX_SEQUENCE_LENGTH - 1, padding="pre"
        )

        # Create input tensor
        input_tensor = tf.convert_to_tensor(padded_tokens, dtype=tf.int32)

        # Generate predictions
        predictions = model(input_tensor)

        # Apply temperature sampling
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

        # Add predicted token to input
        input_tokens.append(predicted_id)

        # Get the corresponding word (ensure it exists in vocabulary)
        if predicted_id < len(tokenizer.index_word):
            predicted_word = tokenizer.index_word.get(predicted_id, "")
            generated_text += " " + predicted_word

            # Stop if EOS token is generated
            if predicted_word == EOS_TOKEN:
                break
        else:
            generated_text += " <UNKNOWN>"

    return generated_text


def main():
    """Main function to train and test the model"""
    # Load and preprocess data - limited to 10,000 recipes
    texts = load_data(max_recipes=5000)  # Further reduced to help with memory

    # Create tokenizer
    tokenizer, vocab_size = create_tokenizer(texts)

    # Build model
    model = build_model(vocab_size)
    model.summary()

    # Create checkpoint callback with the right extension
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    # Train model using batches to avoid memory issues
    print("Training model...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        batch_generator = batch_sequence_generator(
            texts, tokenizer, vocab_size, batch_size=500
        )

        for batch_num, (X_batch, y_batch) in enumerate(batch_generator):
            print(f"  Training batch {batch_num+1}...")
            try:
                # Debug info to verify shapes
                print(f"  X shape: {X_batch.shape}, y shape: {y_batch.shape}")

                # Split into training and validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_batch, y_batch, test_size=0.1, random_state=42
                )

                # Train on this batch
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=1,  # Just one epoch per batch
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    verbose=1,
                )

                # Save checkpoint every 5 batches
                if batch_num % 5 == 0:
                    model.save_weights(f"{checkpoint_dir}/batch_{batch_num}.weights.h5")

                # Clear memory
                tf.keras.backend.clear_session()

                # Limit number of batches to prevent running too long
                if batch_num >= 10:
                    break

            except Exception as e:
                print(f"Error in batch {batch_num+1}: {str(e)}")
                continue

    # Test generation
    print("\nGenerating sample recipes:")

    test_seeds = [
        "Ingredients: ['chicken', 'butter', 'salt', 'pepper', 'garlic'] | Recipe Steps: ",
        "Ingredients: ['flour', 'sugar', 'butter', 'eggs', 'vanilla'] | Recipe Steps: ",
        "Ingredients: ['pasta', 'tomato sauce', 'cheese', 'basil'] | Recipe Steps: ",
    ]

    for seed in test_seeds:
        try:
            generated_recipe = generate_text(model, tokenizer, seed, max_length=100)
            print("\n" + "-" * 40)
            print(generated_recipe)
            print("-" * 40)
        except Exception as e:
            print(f"Error generating text: {str(e)}")

    # Save the model
    try:
        model_save_path = "./recipe_rnn_model.weights.h5"
        model.save_weights(model_save_path)
        print(f"Model weights saved to {model_save_path}")

        # Save the tokenizer
        import pickle

        with open("recipe_tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer saved to recipe_tokenizer.pickle")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


if __name__ == "__main__":
    main()
