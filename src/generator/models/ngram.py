import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Set
import sys
import pickle
import random
import ast
import math
from nltk.util import ngrams


class RecipeNGramModel:
    def __init__(self, n=8, smoothing="laplace", alpha=0.1, random_state=None):
        self.n = n
        self.name = "NGram"
        self.smoothing = smoothing
        self.alpha = alpha
        self.vocabulary: Set[str] = set()
        self.model = {}
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.sep_token = "<STEPS>"

        if n < 2:
            raise ValueError(f"{n} < 2")

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def _build_model(self, tokenized_texts: List[List[str]]) -> Dict:
        print(f"Building {self.n}-gram model on {len(tokenized_texts)} recipes")
        all_order_ngrams = {}

        for i in range(2, self.n + 1):
            all_order_ngrams[i] = []
            for tokens in tokenized_texts:
                all_order_ngrams[i].extend(list(ngrams(tokens, i)))
                self.vocabulary.update(set(tokens))

        models = {}

        for i in range(2, self.n + 1):
            ngram_counts = Counter(all_order_ngrams[i])

            context_counts = Counter(ngram[:-1] for ngram in all_order_ngrams[i])

            probability_model = defaultdict(dict)

            for ngram in ngram_counts:
                context = ngram[:-1]
                word = ngram[-1]

                count = ngram_counts[ngram]
                context_count = context_counts[context]

                if self.smoothing == "laplace":
                    probability = (count + self.alpha) / (
                        context_count + self.alpha * len(self.vocabulary)
                    )
                else:
                    probability = count / context_count if context_count > 0 else 0

                probability_model[context][word] = probability

            models[i] = probability_model
        return models

    def fit(self, ingredients_lists: List[List[str]], steps_lists: List[List[str]]):
        if len(ingredients_lists) != len(steps_lists):
            raise ValueError(
                f"Inconsistent dimensions: {len(ingredients_lists)} ingredients lists vs {len(steps_lists)} steps lists"
            )
        tokenized_texts = [
            [self.start_token]
            + ingredients
            + [self.sep_token]
            + steps
            + [self.end_token]
            for ingredients, steps in zip(ingredients_lists, steps_lists)
        ]
        self.model = self._build_model(tokenized_texts)
        return self

    def _get_probability(self, word: str, context: tuple) -> float:
        context_len = len(context)

        for order in range(min(self.n, context_len + 1), 0, -1):
            current_context = (
                context[-(order - 1) :] if context_len >= (order - 1) else context
            )
            if (
                current_context in self.model[order]
                and word in self.model[order][current_context]
            ):
                return self.model[order][current_context][word]

        if self.smoothing == "laplace":
            return self.alpha / (len(self.vocabulary) * self.alpha)

        return 0

    def predict(self, input_text: List[str], max_steps=30, max_length=200, temperature=None) -> List[str]:
        steps = self._generate_steps(input_text, max_length=max_length)
        return steps[:max_steps]

    def _generate_steps(self, ingredients: List[str], max_length=200) -> List[str]:
        print("Generating recipe steps from ingredients...")
        context = tuple([self.start_token] + ingredients + [self.sep_token])
        if len(context) > self.n - 1:
            context = context[-(self.n - 1) :]

        prediction = []

        for _ in range(max_length):
            next_word = self._generate_next_word(context)
            print(f"Predicted word: ({context}){next_word}")

            if next_word is None or next_word == self.end_token:
                break

            prediction.append(next_word)
            context = (
                tuple(list(context)[1:] + [next_word])
                if len(context) >= self.n - 1
                else tuple(list(context) + [next_word])
            )

        return " ".join(prediction)

    def _generate_next_word(
        self, context: tuple, temperature: float = 1.0
    ) -> Optional[str]:
        context_len = len(context)

        for order in range(
            min(self.n, context_len + 1), 1, -1
        ):  # Changed lower bound to 2
            current_context = (
                context[-(order - 1) :] if context_len >= (order - 1) else context
            )

            if current_context in self.model[order]:
                words_probs = self.model[order][current_context].copy()

                if self.smoothing == "laplace":
                    context_total = sum(words_probs.values())
                    denominator = context_total + self.alpha * len(self.vocabulary)
                    for word in self.vocabulary:
                        if word not in words_probs:
                            words_probs[word] = self.alpha / denominator

                items = list(words_probs.items())
                words = [item[0] for item in items]
                probs = [item[1] for item in items]

                return words[np.argmax(probs)]

        if self.vocabulary:
            return random.choice(
                list(
                    self.vocabulary - {self.start_token, self.end_token, self.sep_token}
                )
            )

        return None

    def evaluate(
        self,
        ingredients_lists: List[List[str]],
        steps_lists: List[List[str]],
        context_size=None,
    ):
        if len(ingredients_lists) != len(steps_lists):
            raise ValueError(
                f"Inconsistent dimensions: {len(ingredients_lists)} ingredients lists vs {len(steps_lists)} steps lists"
            )

        if context_size is None:
            context_size = self.n - 1
        context_size = max(1, min(context_size, self.n - 1))

        tokenized_texts = [
            [self.start_token]
            + ingredients
            + [self.sep_token]
            + steps
            + [self.end_token]
            for ingredients, steps in zip(ingredients_lists, steps_lists)
        ]

        total = 0
        total_valid = 0

        for text in tokenized_texts:
            print(text)
            if len(text) <= context_size + 1:
                continue

            for j in range(len(text) - context_size - 1):
                context = tuple(text[j : j + context_size])
                predicted_word = self._generate_next_word(context)
                actual_word = text[j + context_size]

                if predicted_word == actual_word:
                    total_valid += 1
                total += 1

        if total == 0:
            return 0.0

        return total_valid / total

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)
