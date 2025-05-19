import streamlit as st
from pages.generator_tab import page_generator
from pages.regression_tab import page_regression
from pages.recipe_search_tab import page_recipe_search

def main():
    st.set_page_config(
        page_title="'Los Pollos Hermanos' de Gusteau", page_icon="👨‍🍳"
    )
    st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """, 
    unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        (
            'Recipe Search',
            'Cooking Time Prediction',
            'Auguste Gusteau, The CookBro'
        ),
    )

    if page == 'Cooking Time Prediction':
        page_regression()
    elif page == "Auguste Gusteau, The CookBro":
        page_generator()
    else:
        page_recipe_search()

if __name__ == "__main__":
    main()
