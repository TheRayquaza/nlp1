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

# Download NLTK tokenizers if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def load_model_and_tokenizer(weights_path, tokenizer_path, vocab_size=None):
    """Load the model and tokenizer"""
    # Load tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Determine vocabulary size if not provided
    if vocab_size is None:
        vocab_size = determine_vocab_size(weights_path)
    
    # Build and load the model
    model = build_model(vocab_size)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    model.load_weights(weights_path, skip_mismatch=True)
    
    return model, tokenizer

def preprocess_text(text):
    """Preprocess text for evaluation (lowercase, remove punctuation)"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


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
            "rouge-l": scores["rouge-l"]["f"]
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

def extract_recipe_steps(text):
    """Extract just the recipe steps from the generated text"""
    if '| Recipe Steps:' in text:
        steps = text.split('| Recipe Steps:')[1].strip()
    else:
        steps = text  # Use the whole text if delimiter not found
    
    # Clean the extracted steps
    return steps.strip()

def calculate_ingredient_coverage(ingredients_list, recipe_text):
    """Calculate what percentage of ingredients are mentioned in the recipe"""
    # Clean ingredients (remove quotes etc.)
    clean_ingredients = [ing.strip("' \"").lower() for ing in ingredients_list]
    
    # Count ingredients that appear in generated text
    generated_lower = recipe_text.lower()
    mentioned_ingredients = sum(1 for ing in clean_ingredients if ing in generated_lower)
    
    # Calculate coverage
    coverage = mentioned_ingredients / len(clean_ingredients) if clean_ingredients else 0
    
    return coverage

def format_ingredients_for_model(ingredients):
    """Format ingredients as expected by the model"""
    if isinstance(ingredients, str):
        # If comma-separated string, split into list
        ingredients_list = [ing.strip() for ing in ingredients.split(',')]
    else:
        # Already a list
        ingredients_list = ingredients
    
    # Format as quoted items in brackets
    return "[" + ", ".join(f"'{ing}'" for ing in ingredients_list) + "]"

def load_test_data(file_path):
    """Load test data from CSV or JSON file"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        test_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_data.append(row)
        return test_data
    
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
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
            ingredients = item.get('ingredients', '')
            reference = item.get('recipe', '')
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
        
        # Clean and extract just the recipe steps
        generated_steps = extract_recipe_steps(generated_text)
        reference_steps = reference if reference else ""
        
        # Calculate metrics
        bleu_scores = calculate_bleu(reference_steps, generated_steps)
        rouge_scores = calculate_rouge(reference_steps, generated_steps)
        
        # Calculate ingredient coverage
        if isinstance(ingredients, str):
            ingr_list = [i.strip() for i in ingredients.split(',')]
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
                "ingredient_coverage": coverage
            }
        }
        results.append(result)
    
    return results

def generate_summary_metrics(results):
    """Generate summary metrics across all results"""
    if not results:
        return {}
    
    # Initialize metric accumulators
    metrics = {
        "bleu-1": 0, "bleu-2": 0, "bleu-3": 0, "bleu-4": 0,
        "rouge-1": 0, "rouge-2": 0, "rouge-l": 0,
        "ingredient_coverage": 0
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
    parser = argparse.ArgumentParser(description='Evaluate Recipe Generator using BLEU/ROUGE')
    parser.add_argument('--weights', default='recipe_rnn_model.weights.h5', 
                        help='Path to the model weights file')
    parser.add_argument('--tokenizer', default='recipe_tokenizer.pickle',
                        help='Path to the tokenizer pickle file')
    parser.add_argument('--test_data', required=True,
                        help='Path to test data file (CSV or JSON)')
    parser.add_argument('--output', default='evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Override the vocabulary size')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='Limit number of samples to evaluate (for testing)')
    args = parser.parse_args()

    # Force CPU to avoid GPU memory issues
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    try:
        # Check if files exist
        if not os.path.exists(args.weights):
            print(f"Error: Weights file {args.weights} not found!")
            return 1
            
        if not os.path.exists(args.tokenizer):
            print(f"Error: Tokenizer file {args.tokenizer} not found!")
            return 1
            
        if not os.path.exists(args.test_data):
            print(f"Error: Test data file {args.test_data} not found!")
            return 1
        
        # Load model and tokenizer
        print(f"Loading model from {args.weights}")
        model, tokenizer = load_model_and_tokenizer(
            args.weights, args.tokenizer, args.vocab_size
        )
        
        # Load test data
        print(f"Loading test data from {args.test_data}")
        test_data = load_test_data(args.test_data)
        
        if args.sample_limit and args.sample_limit < len(test_data):
            print(f"Limiting evaluation to {args.sample_limit} samples")
            test_data = test_data[:args.sample_limit]
        
        print(f"Evaluating model on {len(test_data)} test samples...")
        results = evaluate_model(
            model, tokenizer, test_data, 
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Calculate summary metrics
        summary = generate_summary_metrics(results)
        
        # Prepare output
        output = {
            "summary": summary,
            "individual_results": results
        }
        
        # Save results
        print(f"Saving evaluation results to {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        # Display summary metrics
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total samples evaluated: {len(results)}")
        print("\nBLEU Scores:")
        print(f"  BLEU-1: {summary['bleu-1']:.4f}")
        print(f"  BLEU-2: {summary['bleu-2']:.4f}")
        print(f"  BLEU-3: {summary['bleu-3']:.4f}")
        print(f"  BLEU-4: {summary['bleu-4']:.4f}")
        
        print("\nROUGE Scores:")
        print(f"  ROUGE-1: {summary['rouge-1']:.4f}")
        print(f"  ROUGE-2: {summary['rouge-2']:.4f}")
        print(f"  ROUGE-L: {summary['rouge-l']:.4f}")
        
        print(f"\nIngredient Coverage: {summary['ingredient_coverage']:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
