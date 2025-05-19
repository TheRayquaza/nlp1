import base64

def format_recipe_output(generated_text):
    if "Recipe Steps:" in generated_text:
        recipe_part = generated_text.split("Recipe Steps:")[1].strip()
    else:
        recipe_part = generated_text.strip()

    recipe_part = recipe_part.replace("<EOS>", "").replace("<UNKNOWN>", "").strip()

    steps = []
    current_step = ""

    for part in recipe_part.split("."):
        if part.strip():
            current_step += part.strip() + "."
            if len(current_step) > 20:
                steps.append(current_step.strip())
                current_step = ""

    if current_step.strip():
        steps.append(current_step.strip())

    formatted_recipe = ""
    for i, step in enumerate(steps, 1):
        formatted_recipe += f"{i}. {step}\n\n"

    return formatted_recipe


def get_base64_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
