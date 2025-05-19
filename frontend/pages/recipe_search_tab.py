import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
import os

def page_recipe_search():
    st.title("🔍 Recipe Search")
    st.markdown("*Find delicious recipes using ingredients you already have at home!*")

    data_path = 'data/processed/preprocessed_recipe.csv'

    if 'recipe_data' not in st.session_state:
        with st.spinner("Loading recipe data..."):
            if os.path.isfile(data_path):
                data = pd.read_csv(data_path)
                st.session_state['recipe_data'] = data
            else:
                st.error("Recipe data file not found. Please run preprocessing first.")
                return
    else:
        data = st.session_state['recipe_data']
            
    st.sidebar.markdown("### 📌 Recipe Filters")
    
    st.sidebar.markdown("#### Cooking Time")
    time_ranges = ["Any", "Quick (< 30 min)", "Medium (30-60 min)", "Long (> 60 min)"]
    selected_time = st.sidebar.selectbox("Cooking time:", time_ranges)
    
    st.sidebar.markdown("#### Calories")
    cal_min, cal_max = st.sidebar.slider(
        "Calorie range:", 
        int(data['calories'].min()), 
        int(data['calories'].max()),
        [int(data['calories'].min()), int(data['calories'].max() * 0.7)]
    )
    
    st.sidebar.markdown("#### Recipe Tags")
    
    all_tags = []
    for tags_str in data['tags_text'].dropna():
        if isinstance(tags_str, str):
            cleaned_tags = tags_str.replace('[', '').replace(']', '').replace("'", "")
            tags = [tag.strip() for tag in cleaned_tags.split(',')]
            all_tags.extend(tags)
    
    tag_counter = Counter(all_tags)
    top_tags = [tag for tag, count in tag_counter.most_common(15)]
    
    selected_tags = st.sidebar.multiselect(
        "Filter by tags:",
        options=top_tags
    )
    
    st.markdown("## Search Recipes by Ingredients")
    
    st.markdown("Enter ingredients you have (separated by commas):")
    ingredients_input = st.text_input("", placeholder="e.g., chicken, onion, garlic")
    
    col1, col2 = st.columns(2)
    with col1:
        match_type = st.radio(
            "Match type:",
            ["Contains any ingredient", "Contains all ingredients"]
        )
    
    with col2:
        exclude_ingredients = st.text_input(
            "Exclude ingredients (separated by commas):",
            placeholder="e.g., nuts, shellfish"
        )
    
    include_ingredients = [ing.strip().lower() for ing in ingredients_input.split(',') if ing.strip()]
    exclude_list = [ing.strip().lower() for ing in exclude_ingredients.split(',') if ing.strip()]
    
    search_clicked = st.button("🔍 Search Recipes")
    
    if search_clicked or 'search_results' in st.session_state:
        if search_clicked:
            filtered_data = data.copy()
            
            if selected_time == "Quick (< 30 min)":
                filtered_data = filtered_data[filtered_data['minutes'] < 30]
            elif selected_time == "Medium (30-60 min)":
                filtered_data = filtered_data[(filtered_data['minutes'] >= 30) & (filtered_data['minutes'] <= 60)]
            elif selected_time == "Long (> 60 min)":
                filtered_data = filtered_data[filtered_data['minutes'] > 60]
            
            filtered_data = filtered_data[(filtered_data['calories'] >= cal_min) & (filtered_data['calories'] <= cal_max)]
            
            if selected_tags:
                tag_mask = filtered_data['tags_text'].apply(
                    lambda tags_str: any(tag in str(tags_str).lower() for tag in selected_tags) if pd.notna(tags_str) else False
                )
                filtered_data = filtered_data[tag_mask]
            
            if include_ingredients:
                def ingredients_match(ingredients_text, include_list, match_all=False):
                    if pd.isna(ingredients_text):
                        return False
                    
                    ingredients_text = str(ingredients_text).lower()
                    
                    if match_all:
                        return all(ing in ingredients_text for ing in include_list)
                    else:
                        return any(ing in ingredients_text for ing in include_list)
            
                match_condition = match_type == "Contains all ingredients"
                filtered_data = filtered_data[filtered_data['ingredients_text'].apply(
                    lambda x: ingredients_match(x, include_ingredients, match_condition)
                )]
            
            if exclude_list:
                filtered_data = filtered_data[~filtered_data['ingredients_text'].apply(
                    lambda x: ingredients_match(x, exclude_list, False) if pd.notna(x) else False
                )]
            
            st.session_state['search_results'] = filtered_data
            st.session_state['search_params'] = {
                'include': include_ingredients,
                'exclude': exclude_list,
                'time': selected_time,
                'calories': (cal_min, cal_max),
                'tags': selected_tags
            }
        
        results = st.session_state['search_results']
        search_params = st.session_state.get('search_params', {})
        
        if len(results) == 0:
            st.warning("No recipes found matching your criteria. Try broadening your search.")
        else:
            st.success(f"Found {len(results)} recipes matching your criteria!")
            
            st.markdown("### Search Criteria")
            criteria_cols = st.columns(3)
            with criteria_cols[0]:
                if search_params.get('include'):
                    st.markdown(f"**Including:** {', '.join(search_params['include'])}")
                if search_params.get('exclude'):
                    st.markdown(f"**Excluding:** {', '.join(search_params['exclude'])}")
            
            with criteria_cols[1]:
                if search_params.get('time'):
                    st.markdown(f"**Time:** {search_params['time']}")
                if search_params.get('calories'):
                    st.markdown(f"**Calories:** {search_params['calories'][0]} - {search_params['calories'][1]}")
            
            with criteria_cols[2]:
                if search_params.get('tags'):
                    st.markdown(f"**Tags:** {', '.join(search_params['tags'])}")
            
            st.markdown("### Recipe Statistics")
            
            viz_cols = st.columns(2)
            
            with viz_cols[0]:
                fig_time = px.histogram(
                    results,
                    x="minutes",
                    nbins=20,
                    title="Distribution of Cooking Times",
                    labels={"minutes": "Cooking Time (minutes)"},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig_time.update_layout(height=300)
                st.plotly_chart(fig_time, use_container_width=True)
            
            with viz_cols[1]:
                fig_cal = px.histogram(
                    results,
                    x="calories",
                    nbins=20,
                    title="Distribution of Calories",
                    labels={"calories": "Calories"},
                    color_discrete_sequence=['#4ECDC4']
                )
                fig_cal.update_layout(height=300)
                st.plotly_chart(fig_cal, use_container_width=True)
            
            st.markdown("### Matching Recipes")
            
            sort_col, display_col = st.columns([1, 2])
            with sort_col:
                sort_option = st.selectbox(
                    "Sort by:",
                    ["Relevance", "Cooking Time (Low to High)", "Cooking Time (High to Low)", 
                     "Calories (Low to High)", "Calories (High to Low)"]
                )
            
            if sort_option == "Cooking Time (Low to High)":
                results = results.sort_values('minutes')
            elif sort_option == "Cooking Time (High to Low)":
                results = results.sort_values('minutes', ascending=False)
            elif sort_option == "Calories (Low to High)":
                results = results.sort_values('calories')
            elif sort_option == "Calories (High to Low)":
                results = results.sort_values('calories', ascending=False)
            
            with display_col:
                display_format = st.radio(
                    "Display format:",
                    ["Card view", "Table view"],
                    horizontal=True
                )
            
            if display_format == "Table view":
                display_cols = ["name", "minutes", "n_ingredients", "calories", "total_fat", "sugar", "protein"]
                st.dataframe(
                    results[display_cols].style.format({
                        "minutes": "{:.0f}",
                        "calories": "{:.0f}",
                        "total_fat": "{:.1f}",
                        "sugar": "{:.1f}",
                        "protein": "{:.1f}"
                    }),
                    column_config={
                        "name": "Recipe Name",
                        "minutes": "Cook Time (min)",
                        "n_ingredients": "# Ingredients",
                        "calories": "Calories",
                        "total_fat": "Total Fat (g)",
                        "sugar": "Sugar (g)",
                        "protein": "Protein (g)"
                    },
                    use_container_width=True,
                    height=400
                )
                
                selected_recipe_name = st.selectbox(
                    "Select a recipe to view details:",
                    options=results["name"].tolist()
                )
                
                if selected_recipe_name:
                    display_recipe_details(results[results['name'] == selected_recipe_name].iloc[0])
                
            else:
                recipes_to_show = min(20, len(results))
                st.write(f"Showing top {recipes_to_show} recipes")
                
                for i in range(0, recipes_to_show, 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < recipes_to_show:
                            recipe = results.iloc[i + j]
                            with cols[j]:
                                with st.container(border=True):
                                    st.markdown(f"### {recipe['name']}")
                                    
                                    st.markdown(f"⏱️ **Time:** {int(recipe['minutes'])} minutes • 🥣 **Ingredients:** {int(recipe['n_ingredients'])}")
                                    st.markdown(f"🔥 **Calories:** {int(recipe['calories'])} • 🍽️ **Nutrition:** {recipe.get('protein', 0):.1f}g protein")
                                    
                                    if isinstance(recipe.get('description'), str):
                                        desc = recipe['description'][:100] + ('...' if len(recipe['description']) > 100 else '')
                                        st.markdown(f"*{desc}*")
                                    
                                    if st.button("View Recipe Details", key=f"view_{i}_{j}"):
                                        st.session_state['selected_recipe_details'] = recipe
                
                if 'selected_recipe_details' in st.session_state:
                    display_recipe_details(st.session_state['selected_recipe_details'])

def display_recipe_details(recipe):
    """Display detailed information about a selected recipe"""
    st.markdown("---")
    st.markdown(f"## {recipe['name']}")
    
    # Recipe metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cooking Time", f"{int(recipe['minutes'])} min")
    with col2:
        st.metric("Ingredients", f"{int(recipe['n_ingredients'])}")
    with col3:
        st.metric("Steps", f"{int(recipe['n_steps'])}")
    
    # Detailed information
    tabs = st.tabs(["Overview", "Ingredients", "Nutrition", "Steps"])
    
    with tabs[0]:
        # Overview tab
        if isinstance(recipe.get('description'), str):
            st.markdown("### Description")
            st.markdown(recipe['description'])
        
        # Tags
        if isinstance(recipe.get('tags_text'), str) and recipe['tags_text']:
            st.markdown("### Tags")
            
            # Clean and extract tags
            cleaned_tags = recipe['tags_text'].replace('[', '').replace(']', '').replace("'", "")
            tags = [tag.strip() for tag in cleaned_tags.split(',') if tag.strip()]
            
            # Display tags with black text
            tags_html = ' '.join([
                f'<span style="background-color: #f0f2f6; color: black; padding: 5px 10px; margin: 2px; border-radius: 15px; font-size: 0.8em;">{tag}</span>'
                for tag in tags
            ])
            st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### Ingredients")

        if isinstance(recipe.get('ingredients_text'), str):
            cleaned_text = recipe['ingredients_text'].replace('[', '').replace(']', '').replace("'", "")
            ingredients = [i.strip() for i in cleaned_text.split(',') if i.strip()]

            copy_text = "\n".join(ingredients)
            st.text_area("📋 Copy Ingredients", copy_text, height=150)

            if st.button("🛒 Generate recipe with these ingredients"):
                st.session_state.ingredients_input = ", ".join(ingredients)
                st.switch_page("pages/generator_tab.py")

    with tabs[2]:
        st.markdown("### Nutrition Information")
        
        nutrition_cols = {
            'calories': 'Calories', 
            'total_fat': 'Total Fat (g)', 
            'sugar': 'Sugar (g)', 
            'sodium': 'Sodium (mg)', 
            'protein': 'Protein (g)', 
            'saturated_fat': 'Saturated Fat (g)', 
            'carbohydrates': 'Carbohydrates (g)'
        }
        
        nutrition_data = {}
        for col, label in nutrition_cols.items():
            if col in recipe and not pd.isna(recipe[col]):
                nutrition_data[label] = recipe[col]
        
        if nutrition_data:
            nutrition_df = pd.DataFrame({
                'Nutrient': list(nutrition_data.keys()),
                'Value': list(nutrition_data.values())
            })
            
            if 'Calories' in nutrition_data:
                calories_df = nutrition_df[nutrition_df['Nutrient'] == 'Calories']
                other_nutrients_df = nutrition_df[nutrition_df['Nutrient'] != 'Calories']
                
                st.metric("Calories", f"{int(nutrition_data['Calories'])}")
                
                if not other_nutrients_df.empty:
                    fig = px.bar(
                        other_nutrients_df,
                        x='Nutrient',
                        y='Value',
                        color='Nutrient',
                        title='Nutritional Composition',
                        labels={'Value': 'Amount (g/mg)'},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(
                    nutrition_df,
                    x='Nutrient',
                    y='Value',
                    color='Nutrient',
                    title='Nutritional Composition',
                    labels={'Value': 'Amount'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            nutrition_table = pd.DataFrame({
                'Nutrient': list(nutrition_data.keys()),
                'Value': list(nutrition_data.values())
            })
            st.dataframe(
                nutrition_table,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No nutrition information available for this recipe.")
    
    with tabs[3]:
        # Steps tab
        st.markdown("### Cooking Instructions")
        
        if 'steps_string_standardize' in recipe and isinstance(recipe['steps_string_standardize'], str):
            steps_text = recipe['steps_string_standardize']
            
            if re.search(r'^\d+\.', steps_text):
                steps = re.split(r'\d+\.', steps_text)
                steps = [step.strip() for step in steps if step.strip()]
            else:
                steps = [step.strip() + '.' for step in steps_text.split('.') if step.strip()]
            
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}:** {step}")
        else:
            st.info("No cooking instructions available for this recipe.")
    
    st.button("Save Recipe", key="save_recipe", help="Save this recipe to your favorites (demo only)")
    
    # Add a button to go back to results
    if st.button("Back to Results"):
        if 'selected_recipe_details' in st.session_state:
            del st.session_state['selected_recipe_details']
        st.experimental_rerun()
