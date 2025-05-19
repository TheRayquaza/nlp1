import streamlit as st
import random
import time
import base64
import tensorflow as tf
import numpy as np
import pickle
import os
import re
import sys
import pathlib
from tensorflow.keras.preprocessing.sequence import pad_sequences

from generator.util import format_recipe_output, get_base64_image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.models.optimized_feedforward_nn import OptimizedBatchFeedForwardNN
from src.generator.models.ngram import RecipeNGramModel
from src.generator.models.gru_rnn import GruRNNModel

GUSTEAU_IMG_PATH = "img/Gusteau.png"
GUSTEAU_2_IMG_PATH = "img/Gusteau_2.png"
USER_IMG_PATH = "img/ratatouille.png"

# Set GPU to not be used
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Function to extract ingredients from user message
def extract_ingredients(message):
    """Extract ingredients from user message"""
    ingredients = ""

    # Pattern matching for various ways users might provide ingredients
    if "ingredients:" in message.lower():
        ingredients = message.lower().split("ingredients:")[1].strip()
    elif "using" in message.lower() and any(
        word in message.lower() for word in ["ingredients", "items", "foods"]
    ):
        ingredients_part = message.lower().split("using")[1]
        # Find the part after "ingredients", "items", or "foods"
        for term in ["ingredients", "items", "foods"]:
            if term in ingredients_part:
                ingredients = ingredients_part.split(term)[1].strip(" :.,")
                break
        if not ingredients:  # If no specific term found
            ingredients = ingredients_part.strip(" :.,")
    elif re.search(r"recipe\s+with\s+([\w\s,]+)", message.lower()):
        # Match "recipe with X, Y, Z"
        match = re.search(r"recipe\s+with\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip()
    elif re.search(r"make.*using\s+([\w\s,]+)", message.lower()):
        # Match "make something using X, Y, Z"
        match = re.search(r"make.*using\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip()
    elif re.search(r"cook.*with\s+([\w\s,]+)", message.lower()):
        # Match "cook something with X, Y, Z"
        match = re.search(r"cook.*with\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip()
    elif "i have" in message.lower() and any(
        word in message.lower() for word in ["recipe", "cook", "make", "dish"]
    ):
        # Match "I have X, Y, Z. What can I make?"
        ingredients_part = message.lower().split("i have")[1].strip()
        # Find where the question starts
        for question_start in ["what", "can", "how", "?"]:
            if question_start in ingredients_part:
                ingredients = ingredients_part.split(question_start)[0].strip(" :.,")
                break
        if not ingredients:  # If no question found
            ingredients = ingredients_part.strip(" :.,")
    elif re.search(r"i have\s+([\w\s,]+)", message.lower()):
        # Simple "I have X, Y, Z"
        match = re.search(r"i have\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip(" :.,?")
    elif any(
        message.lower().startswith(term)
        for term in ["can you", "could you", "please", "help"]
    ):
        # Check if the message contains a comma-separated list anywhere
        if "," in message:
            # Extract the part of the message with commas
            comma_parts = [part for part in re.split(r"[.?!]", message) if "," in part]
            if comma_parts:
                ingredients = comma_parts[0].strip(" :.,?")

    # If still nothing found, check if the message itself looks like a comma-separated list
    if not ingredients and "," in message and len(message.split(",")) >= 2:
        # The message might just be a list of ingredients
        ingredients = message.strip()

    return ingredients


def page_generator():
    st.title("Auguste Gusteau, The CookBro")
    st.markdown(
        "*Try our recipe generator made with Gusteau, The Renowned French Cook, with a 5-Star Michelin Restaurant. Don't forget that Anyone can cook!*"
    )

    @st.cache_resource
    def load_models():
        models = {
            "rnn": GruRNNModel().load_model(
                model_path="./models/saved/generator/recipe_rnn_model.keras",
                tokenizer_path="./models/saved/generator/recipe_tokenizer.pickle",
            ),
            "ffnn": OptimizedBatchFeedForwardNN().load("./models/saved/generator/ffnn.keras"),
            "ngram": RecipeNGramModel().load("./models/saved/generator/ngram2.pkl")
        }
        models["ngram"].name = "NGram" # fixup as model is already loaded in pkl
        return models

    all_models = load_models()
    model_loaded = len(all_models) > 0

    if "rnn" in all_models:
        default_model = "rnn"
    elif "ffnn" in all_models:
        default_model = "ffnn"
    elif "ngram" in all_models:
        default_model = "ngram"
    else:
        default_model = None
        model_loaded = False

    try:
        gusteau_base64 = get_base64_image(GUSTEAU_IMG_PATH)
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{gusteau_base64}" width="600"/>
                <p><em>Auguste Gusteau</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not load main image: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour! I am Auguste Gusteau. Tell me what ingredients you have directly in the chat, and I'll create a wonderful recipe for you! You can also use the form below. Anyone can cook with my help!",
            }
        ]

    st.subheader("Generate a Recipe")

    with st.form("recipe_form"):
        model_options = {
            "rnn": "RNN (GRU) - Most creative",
            "ffnn": "Feed-Forward NN - More structured",
            "ngram": "N-Gram - Closest to original recipes",
        }

        available_models = {k: v for k, v in model_options.items() if k in all_models}

        if available_models:
            selected_model = st.selectbox(
                "Select Recipe Generation Model:",
                options=list(available_models.keys()),
                format_func=lambda x: available_models.get(x, x),
                index=(
                    list(available_models.keys()).index(default_model)
                    if default_model in available_models
                    else 0
                ),
            )
        else:
            selected_model = None
            st.warning("No models available. Please check the model paths.")

        if "ingredients_input" not in st.session_state:
            st.session_state.ingredients_input = ""

        ingredients_input = st.text_input(
            "Enter your ingredients (separated by commas):",
            value=st.session_state.ingredients_input,
            placeholder="chicken, butter, salt, pepper, garlic",
        )

        st.session_state.ingredients_input = ingredients_input

        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider(
                "Recipe Length", min_value=50, max_value=500, value=200, step=10
            )
        with col2:
            temperature = st.slider(
                "Creativity Level", min_value=0.5, max_value=1.5, value=1.0, step=0.1
            )

        generate_button = st.form_submit_button("Generate Recipe")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = selected_model
    else:
        st.session_state.selected_model = selected_model

    if "max_length" not in st.session_state:
        st.session_state.max_length = max_length
    else:
        st.session_state.max_length = max_length

    if "temperature" not in st.session_state:
        st.session_state.temperature = temperature
    else:
        st.session_state.temperature = temperature

    if generate_button and ingredients_input and model_loaded and selected_model:
        with st.spinner(
            f"Chef Gusteau is creating your recipe using the {all_models[selected_model].name} model..."
        ):
            try:
                if selected_model == "ffnn":
                    ingredients_input = [ingredients_input.split(",")]
                if selected_model == "ngram":
                    ingredients_input = ingredients_input.split(",")
                raw_recipe = all_models[selected_model].predict(
                    input_text=ingredients_input,
                    max_length=max_length,
                    temperature=temperature,
                )
                if selected_model == "ffnn":
                    raw_recipe = " ".join(raw_recipe[0])

                formatted_recipe = format_recipe_output(raw_recipe)

                user_message = f"Please create a recipe using these ingredients: {ingredients_input}"
                st.session_state.messages.append(
                    {"role": "user", "content": user_message}
                )

                assistant_response = f"""
## A Recipe by Chef Gusteau
**Using {all_models[selected_model].name} model**

**Ingredients:** {ingredients_input}

### Instructions:
{formatted_recipe}

*Bon appétit! Anyone can cook!*
"""
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )

            except Exception as e:
                st.error(f"Error generating recipe: {e}")

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=GUSTEAU_2_IMG_PATH):
                st.markdown(message["content"])
        else:
            with st.chat_message("user", avatar=USER_IMG_PATH):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask Chef Gusteau anything or provide ingredients..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_IMG_PATH):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=GUSTEAU_2_IMG_PATH):
            message_placeholder = st.empty()
            full_response = ""

            ingredients = prompt

            is_recipe_request = (
                any(
                    word in prompt.lower()
                    for word in ["recipe", "make", "cook", "dish", "prepare", "create"]
                )
                or len(ingredients) > 5
            )

            if (
                ingredients
                and (is_recipe_request or "," in ingredients)
                and model_loaded
                and selected_model
            ):
                try:
                    processing_message = f"Ah! I see you've provided ingredients. Let me create a wonderful recipe for you using my {all_models[selected_model].name} model..."
                    for chunk in processing_message.split():
                        full_response += chunk + " "
                        time.sleep(0.03)
                        message_placeholder.markdown(full_response + "▌")

                    with st.spinner(
                        f"Chef Gusteau is creating your recipe using the {all_models[selected_model].name} model..."
                    ):
                        if selected_model == "ffnn":
                            ingredients_input = [ingredients_input.split(",")]
                        elif selected_model == "ngram":
                            ingredients_input = ingredients_input.split(",")
                        raw_recipe = all_models[selected_model].predict(
                            input_text=ingredients_input,
                            max_length=st.session_state.max_length,
                            temperature=st.session_state.temperature,
                        )
                        if selected_model == "ffnn":
                            raw_recipe = " ".join(raw_recipe)

                        formatted_recipe = format_recipe_output(raw_recipe)

                    assistant_response = f"""
## A Recipe by Chef Gusteau
**Using {all_models[selected_model].name} model**

**Ingredients:** {ingredients}

### Instructions:
{formatted_recipe}

*Bon appétit! Anyone can cook!*
"""
                    message_placeholder.markdown(assistant_response)

                except Exception as e:
                    error_msg = f"Oh non! I had trouble creating a recipe with those ingredients: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    assistant_response = error_msg
            else:
                if (
                    any(
                        word in prompt.lower()
                        for word in ["ingredients", "have", "got", "using"]
                    )
                    and not is_recipe_request
                ):
                    assistant_response = "It sounds like you're telling me about ingredients. If you'd like me to create a recipe, please list your ingredients separated by commas, or clearly ask for a recipe with those ingredients!"
                elif any(
                    word in prompt.lower()
                    for word in ["model", "select", "switch", "change", "ai", "neural"]
                ):
                    available_models_str = ", ".join(
                        [f"{m.name}" for m in all_models.values()]
                    )
                    assistant_response = f"I can generate recipes using different AI models! Currently, I have {available_models_str} available. You can select the model in the dropdown menu above the ingredient input field."
                else:
                    cooking_responses = [
                        "Anyone can cook, but only the fearless can be great!",
                        "The only thing predictable about life is its unpredictability!",
                        "Great cooking is not for the faint of heart, mon ami!",
                        "You must not let anyone define your limits because of where you come from.",
                        "Cooking is like life - you never know what flavors await unless you try!",
                        "In cooking, as in life, you must be bold and adventurous!",
                        "Your ingredients may be simple, but your dish can be extraordinary!",
                        "A recipe is only as good as the chef who interprets it.",
                        "In my kitchen, there are no mistakes, only new discoveries!",
                        "The difference between a good cook and a great one is imagination!",
                    ]

                    if any(
                        word in prompt.lower()
                        for word in ["hello", "hi", "hey", "bonjour"]
                    ):
                        assistant_response = "Bonjour! I am Auguste Gusteau. Tell me what ingredients you have, and I'll create a wonderful recipe for you! Just list your ingredients separated by commas."
                    elif any(
                        word in prompt.lower()
                        for word in ["help", "how", "what can you"]
                    ):
                        assistant_response = "I can help you create delicious recipes! Simply tell me what ingredients you have by listing them separated by commas in the chat, and I'll craft a recipe for you. You can also use the form above to select which AI model to use, adjust recipe length and creativity level."
                    else:
                        assistant_response = random.choice(cooking_responses)
                        assistant_response += "\n\nIf you want me to create a recipe, please list your ingredients separated by commas in the chat, or use the form above to select a model and other settings!"

                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.03)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                assistant_response = full_response

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )


if __name__ == "__main__":
    page_generator()
