#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pickle
import argparse
import os
import csv
import json
import re
from pathlib import Path

# For BLEU score calculation
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# For ROUGE score calculation
from rouge import Rouge

# Import functions from the toto.py script
from toto import build_model, generate_text, clean_recipe_text, determine_vocab_size

# Download NLTK tokenizers if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load a full .keras model and tokenizer"""
    # Load tokenizer
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    # Load full model from .keras file
    model = tf.keras.models.load_model(model_path)

    return model, tokenizer


def preprocess_text(text):
    """Preprocess text for evaluation (lowercase, remove punctuation)"""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s\']", " ", text)

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate texts"""
    # Tokenize sentences to words
    reference_tokens = nltk.word_tokenize(preprocess_text(reference))
    candidate_tokens = nltk.word_tokenize(preprocess_text(candidate))

    # Calculate BLEU scores with smoothing
    smoothing = SmoothingFunction().method1
    bleu_1 = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing,
    )
    bleu_2 = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing,
    )
    bleu_3 = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(0.33, 0.33, 0.33, 0),
        smoothing_function=smoothing,
    )
    bleu_4 = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    return {"bleu-1": bleu_1, "bleu-2": bleu_2, "bleu-3": bleu_3, "bleu-4": bleu_4}


def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores between reference and candidate texts"""
    # Initialize ROUGE scorer
    rouge = Rouge()

    # Ensure both texts have content
    if not reference.strip() or not candidate.strip():
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

    # Calculate ROUGE scores
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}


def extract_recipe_steps(text):
    """Extract just the recipe steps from the generated text"""
    print(f"Extracting recipe steps from text: {text}")
    if "| Recipe Steps:" in text:
        steps = text.split("| Recipe Steps:")[1]
        print(f"Extracted steps: {steps}")
    else:
        steps = text  # Use the whole text if delimiter not found

    # Clean the extracted steps
    return steps


def calculate_ingredient_coverage(ingredients_list, recipe_text):
    """Calculate what percentage of ingredients are mentioned in the recipe"""
    # Clean ingredients (remove quotes etc.)
    clean_ingredients = [ing.strip("' \"").lower() for ing in ingredients_list]

    # Count ingredients that appear in generated text
    generated_lower = recipe_text.lower()
    mentioned_ingredients = sum(
        1 for ing in clean_ingredients if ing in generated_lower
    )

    # Calculate coverage
    coverage = (
        mentioned_ingredients / len(clean_ingredients) if clean_ingredients else 0
    )

    return coverage


def format_ingredients_for_model(ingredients):
    """Format ingredients as expected by the model"""
    if isinstance(ingredients, str):
        # If comma-separated string, split into list
        ingredients_list = [ing.strip() for ing in ingredients.split(",")]
    else:
        # Already a list
        ingredients_list = ingredients

    # Format as quoted items in brackets
    return "[" + ", ".join(f"'{ing}'" for ing in ingredients_list) + "]"


def load_test_data(file_path):
    """Load test data from CSV or JSON file"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        test_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_data.append(row)
        return test_data

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def evaluate_model(model, tokenizer, test_data, max_length=100, temperature=1.0):
    """Evaluate model on test data and calculate metrics"""
    results = []

    for item in test_data:
        # Extract ingredients and reference recipe
        if isinstance(item, dict):
            # Handle dictionary format
            ingredients = item.get("ingredients", "")
            reference = item.get("recipe", "")
        elif isinstance(item, (list, tuple)):
            # Handle list format [ingredients, recipe]
            if len(item) >= 2:
                ingredients, reference = item[0], item[1]
            else:
                continue
        else:
            # Skip invalid items
            continue

        # Format ingredients for the model input
        if isinstance(ingredients, list):
            formatted_ingredients = format_ingredients_for_model(ingredients)
        else:
            formatted_ingredients = format_ingredients_for_model(ingredients)

        # Create seed text
        seed_text = f"Ingredients: {formatted_ingredients} | Recipe Steps: "

        # Generate recipe
        generated_text = generate_text(
            model, tokenizer, seed_text, max_length=max_length, temperature=temperature
        )

        print(f"Generated Recipe: {generated_text.split}")

        # Clean and extract just the recipe steps
        generated_steps = extract_recipe_steps(generated_text)
        reference_steps = reference if reference else ""

        # Calculate metrics
        bleu_scores = calculate_bleu(reference_steps, generated_steps)
        rouge_scores = calculate_rouge(reference_steps, generated_steps)

        # Calculate ingredient coverage
        if isinstance(ingredients, str):
            ingr_list = [i.strip() for i in ingredients.split(",")]
        else:
            ingr_list = ingredients
        coverage = calculate_ingredient_coverage(ingr_list, generated_steps)

        # Store results
        result = {
            "ingredients": ingredients,
            "reference_recipe": reference_steps,
            "generated_recipe": generated_steps,
            "metrics": {
                "bleu": bleu_scores,
                "rouge": rouge_scores,
                "ingredient_coverage": coverage,
            },
        }
        results.append(result)

    return results


def generate_summary_metrics(results):
    """Generate summary metrics across all results"""
    if not results:
        return {}

    # Initialize metric accumulators
    metrics = {
        "bleu-1": 0,
        "bleu-2": 0,
        "bleu-3": 0,
        "bleu-4": 0,
        "rouge-1": 0,
        "rouge-2": 0,
        "rouge-l": 0,
        "ingredient_coverage": 0,
    }

    # Sum all metrics
    for result in results:
        result_metrics = result["metrics"]
        metrics["bleu-1"] += result_metrics["bleu"]["bleu-1"]
        metrics["bleu-2"] += result_metrics["bleu"]["bleu-2"]
        metrics["bleu-3"] += result_metrics["bleu"]["bleu-3"]
        metrics["bleu-4"] += result_metrics["bleu"]["bleu-4"]

        metrics["rouge-1"] += result_metrics["rouge"]["rouge-1"]
        metrics["rouge-2"] += result_metrics["rouge"]["rouge-2"]
        metrics["rouge-l"] += result_metrics["rouge"]["rouge-l"]

        metrics["ingredient_coverage"] += result_metrics["ingredient_coverage"]

    # Calculate averages
    count = len(results)
    for key in metrics:
        metrics[key] /= count

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Recipe Generator using BLEU/ROUGE"
    )
    parser.add_argument(
        "--tokenizer",
        default="recipe_tokenizer.pickle",
        help="Path to the tokenizer pickle file",
    )
    parser.add_argument(
        "--test_data", required=True, help="Path to test data file (CSV or JSON)"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=None, help="Override the vocabulary size"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--model", default="recipe_rnn_model.keras", help="recipe_rnn_model.keras"
    )
    args = parser.parse_args()

    # Force CPU to avoid GPU memory issues
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        # Replace usage of args.weights with args.model:
        model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizer)

        # Load test data
        print(f"Loading test data from {args.test_data}")
        test_data = load_test_data(args.test_data)

        if args.sample_limit and args.sample_limit < len(test_data):
            print(f"Limiting evaluation to {args.sample_limit} samples")
            test_data = test_data[: args.sample_limit]

        print(f"Evaluating model on {len(test_data)} test samples...")
        results = evaluate_model(
            model,
            tokenizer,
            test_data,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        print(results)
        return 0

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
