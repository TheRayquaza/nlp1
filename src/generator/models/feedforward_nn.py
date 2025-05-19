import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import time

class BatchFeedForwardNN:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None
        self.context = None
        self.batch_size = 32

    def _build_vocabulary(self, ingredients: List[List[str]], steps: List[List[str]]) -> None:
        print("Building vocabulary...")
        all_tokens = [token for lst in ingredients + steps for token in lst] + ['<UNKNOWN>', '<s>', '</s>', "<STEPS>"]
        counter = Counter(all_tokens)
        self.vocab = [token for token, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def _prepare_train(self, ingredients: List[List[str]], steps: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:        
        if self.vocab is None:
            self._build_vocabulary(ingredients, steps)

        context_targets = defaultdict(Counter)
        
        for i in range(len(ingredients)):
            ingredient_ids = [self.token_to_id["<s>"]] + \
                             [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) for token in ingredients[i]] + \
                             [self.token_to_id['<STEPS>']]
            step_ids = [self.token_to_id.get(step_token, self.token_to_id['<UNKNOWN>']) for step_token in steps[i]] + \
                       [self.token_to_id["</s>"]]

            tokens_ids = ingredient_ids + step_ids

            for k in range(len(tokens_ids) - self.context):
                context_window = tuple(tokens_ids[k:k+self.context])
                context_targets[context_window][tokens_ids[k+self.context]] += 1

        print(f"Found {len(context_targets)} unique context-grams")

        X_data = []
        y_data = []
        sample_weights = []

        print("Building sample weights using context gram...")
        for context_gram, target_counts in context_targets.items():
            total_count = sum(target_counts.values())
            for target_id, count in target_counts.items():
                X_data.append(list(context_gram))
                y_data.append(target_id)
                sample_weights.append(count / total_count)

        return np.array(X_data, dtype=np.int32), np.array(y_data, dtype=np.int32), np.array(sample_weights, dtype=np.float32)
        
    def fit(self, ingredients: List[List[str]], steps: List[List[str]],
            embedding_dim=256, hidden_dim=512, context=3,
            epochs=10, batch_size=32, validation_split=0.1,
            learning_rate=1e-2
    ):
        if len(ingredients) != len(steps):
            raise ValueError(f"dimension mismatch {len(ingredients)} vs {len(steps)}")

        start_time = time.time()
        self.batch_size = batch_size
        self.context = context

        print("Preparing unique context-grams for training...")
        X_train, y_train, sample_weights = self._prepare_train(ingredients, steps)

        vocab_size = len(self.vocab)
        print(f"Vocab size: {vocab_size}")
        print(f"Unique training samples: {len(X_train)}")
        print(f"Data preparation took {time.time() - start_time:.2f} seconds")
        for i in range(5):
            print(f"Sample {i}: {X_train[i]} - {y_train[i]}")

        print("Building model...")
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=(context,), dtype=tf.int32))
        self.model.add(keras.layers.Embedding(vocab_size * context, embedding_dim, name="embedding"))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(hidden_dim, activation='relu', name="hidden"))
        self.model.add(keras.layers.Dense(vocab_size, activation='softmax', name="output"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        self.model.build(input_shape=(None, context))

        print(self.model.summary())
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train, sample_weights)
        ).shuffle(buffer_size=10000)
        
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size
        
        train_ds = train_dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = train_dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        start_train_time = time.time()
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks= [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )

        print(f"Training took {time.time() - start_train_time:.2f} seconds")
        print(f"Total process took {time.time() - start_time:.2f} seconds")

        return self

    def _prepare_test(self, ingredients: List[List[str]]):
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Fit the model first!")
        
        all_contexts = []
        for ingredient_list in ingredients:
            ingredient_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) 
                              for token in ingredient_list] + [self.token_to_id['<STEPS>']]
                
            if len(ingredient_ids) >= self.context:
                context_tokens = ingredient_ids[-self.context:]
            else:
                padding = [self.token_to_id['<UNKNOWN>']] * (self.context - len(ingredient_ids))
                context_tokens = padding + ingredient_ids
            
            all_contexts.append(context_tokens)
        
        return all_contexts
    
    def predict(self, input_text: List[List[str]], max_length=20, temperature=None) -> List[List[str]]:
        if self.model is None:
            raise ValueError("Model not fitted!")
        
        all_contexts = self._prepare_test(input_text)
        
        num_ingredients = len(input_text)
        current_tokens = [['<s>'] for _ in range(num_ingredients)]
        completed = [False] * num_ingredients

        for step in range(max_length):
            if all(completed):
                break
            
            active_indices = [i for i, is_complete in enumerate(completed) if not is_complete]
            
            for batch_start in range(0, len(active_indices), self.batch_size):
                batch_indices = active_indices[batch_start:batch_start + self.batch_size]
                
                batch_contexts = []
                for idx in batch_indices:
                    if step == 0:
                        context = all_contexts[idx]
                    else:
                        predicted_tokens = current_tokens[idx][1:]
                        
                        if len(predicted_tokens) >= self.context:
                            token_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) 
                                       for token in predicted_tokens[-self.context:]]
                        else:
                            orig_needed = self.context - len(predicted_tokens)
                            orig_context = all_contexts[idx][:orig_needed]
                            
                            # Convert predicted tokens to IDs
                            pred_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) 
                                      for token in predicted_tokens]
                            
                            # Combine original and predicted
                            token_ids = orig_context + pred_ids
                        
                        context = token_ids
                
                    batch_contexts.append(self.context)
                
                batch_input = np.array(batch_contexts)

                expected_shape = (len(batch_indices), self.context)
                if batch_input.shape != expected_shape:
                    if len(batch_input.shape) == 2 and batch_input.shape[1] == self.context:
                        pass
                    else:
                        try:
                            batch_input = batch_input.reshape(expected_shape)
                        except ValueError:
                            raise ValueError(f"Cannot reshape batch input from {batch_input.shape} to {expected_shape}")
                
                predictions = self.model.predict_on_batch(batch_input)
                
                for i, pred_idx in enumerate(batch_indices):
                    prediction = predictions[i] if len(predictions) > 1 else predictions[0]
                    
                    next_token_id = np.argmax(prediction)
                    next_token = self.id_to_token[next_token_id]
                    
                    current_tokens[pred_idx].append(next_token)
                    
                    if next_token == '</s>':
                        completed[pred_idx] = True
        
        all_predictions = []
        for tokens in current_tokens:
            if tokens[0] == '<s>':
                tokens = tokens[1:]
            
            if tokens and tokens[-1] == '</s>':
                tokens = tokens[:-1]
                
            all_predictions.append(tokens)
        
        return all_predictions

    def evaluate(self, ingredients: List[List[str]], steps: List[List[str]]):
        if self.model is None:
            raise ValueError("Model not fitted!")

        X, y, weights = self._prepare_train(ingredients, steps)

        ds = tf.data.Dataset.from_tensor_slices(
            (X, y, weights)
        ).shuffle(buffer_size=10000)
        ds = ds.batch(self.batch_size)

        return self.model.evaluate(ds)

    def save(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save!")

        self.model.save(filepath)

        np.savez(f"{filepath}_vocab.npz", 
                 vocab=np.array(self.vocab, dtype=object),
                 context=np.array([self.context]),
                 batch_size=np.array([self.batch_size]),
        )

    def load(self, filepath: str):
        self.model = keras.models.load_model(filepath)

        data = np.load(f"{filepath}_vocab.npz", allow_pickle=True)

        self.vocab = data['vocab'].tolist()
        self.context = int(data['context'][0])
        self.batch_size = int(data['batch_size'][0])
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        return self
