import streamlit as st
import pandas as pd
import os
import numpy as np

from frontend.regression.train_models import select_model, create_feature_matrices, train_test_split_custom, predict_cooking_time
from frontend.regression.preprocessing import process_data

models_dict = {
    "Linear Regression - TF-IDF": "lr_tf",
    "Bayesian Ridge - TF-IDF": "bayesian_tf",
    "Feedforward Neural Network - TF-IDF": "forward_nn_tf",
    #"Gated Recurrent Unit - TF-IDF": "gru_tf",
    "Linear Regression - Word2Vec": "lr_w2v",
    "Bayesian Ridge - Word2Vec": "bayesian_w2v",
    "Feedforward Neural Network - Word2Vec": "forward_nn_w2v",
    #"Gated Recurrent Unit - Word2Vec": "gru_w2v",
}


def page_regression():
    st.title("⏱️ Time Cooking Predictor")
    st.markdown("*Never lose track of time with our Time Cooking Predictor !*")

    # Initialize session state for tracking app state
    if 'models_computed' not in st.session_state:
        st.session_state['models_computed'] = False
    
    if 'show_predictions' not in st.session_state:
        st.session_state['show_predictions'] = False
        
    if 'previous_model_selection' not in st.session_state:
        st.session_state['previous_model_selection'] = []

    # Sidebar model selection
    st.sidebar.markdown("### 📌 Models")
    selected_models = {}
    for model_name in models_dict.keys():
        selected_models[model_name] = st.sidebar.checkbox(f"{model_name}")

    selected_model_names = [
        name for name, selected in selected_models.items() if selected
    ]

    if not selected_model_names:
        st.sidebar.warning("Please select at least one model for comparison")
        return
    
    # Check if model selection has changed
    if st.session_state['models_computed'] and set(selected_model_names) != set(st.session_state.get('previous_model_selection', [])):
        st.warning("Model selection has changed. You need to recompute predictions.")
        st.session_state['models_computed'] = False
    
    # Compute button - show if models haven't been computed or selection changed
    compute_clicked = st.button("🚀 Compute Predictions")
    
    if compute_clicked:
        data_path = 'data/processed/preprocessed_recipe.csv'
        st.session_state["metrics_displayed"] = False
        # Use with statement for preprocessing status message
        with st.status("Preparing data...") as status:
            if os.path.isfile(data_path):
                status.update(label="Loading preprocessed data...")
                data = pd.read_csv(data_path)
                status.update(label="Data loaded successfully!", state="complete")
            else:
                status.update(label="Preprocessed file not found. Running preprocessing...")
                data = process_data(save=True)
                status.update(label="Preprocessing completed successfully!", state="complete")
        
        # Create feature matrices outside the with statement since we're done with preprocessing
        with st.spinner("Creating feature matrices..."):
            feature_matrices_tf = create_feature_matrices(data, 'tf')
            feature_matrices_w2v = create_feature_matrices(data, 'w2v')
            
            X_train_tf, X_test_tf, y_train_tf, y_test_tf, X_train_indices_tf, X_test_indices_tf = train_test_split_custom(data, feature_matrices_tf)
            X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v, X_train_indices_w2v, X_test_indices_w2v = train_test_split_custom(data, feature_matrices_w2v)

        df_results = pd.DataFrame()
        
        # Progress bar for model calculation
        progress_bar = st.progress(0)
        total_models = len(selected_model_names)
        
        # Process each selected model
        for i, model_name in enumerate(selected_model_names):
            # Create a spinner for each model
            with st.spinner(f"Calculating {model_name}..."):
                model_key = models_dict[model_name]
                
                if model_key.endswith('tf'):
                    results = select_model(model_key, X_train_tf, X_test_tf, y_train_tf, y_test_tf)
                elif model_key.endswith('w2v'):
                    results = select_model(model_key, X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v)
                else:
                    st.error(f"Unknown model type: {model_key}")
                    continue
                
                results["Model"] = model_name
                df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index=True)
                
                # Update progress
                progress_bar.progress((i + 1) / total_models)
        
        # Clear progress bar when done
        progress_bar.empty()
        
        # Display results
        if not df_results.empty and not st.session_state.get("metrics_displayed", False):
            st.session_state['feature_matrices'] = {
                'tf': feature_matrices_tf,
                'w2v': feature_matrices_w2v
            }
            st.session_state['train_test_data'] = {
                'tf': (X_train_tf, X_test_tf, y_train_tf, y_test_tf, X_train_indices_tf, X_test_indices_tf),
                'w2v': (X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v, X_train_indices_w2v, X_test_indices_w2v)
            }
            st.session_state['raw_data'] = data
            st.session_state['selected_models'] = selected_model_names
            st.session_state['previous_model_selection'] = selected_model_names.copy()
            st.session_state['models_computed'] = True
            st.session_state['df_results'] = df_results

        else:
            st.warning("No model results to display. There might have been an error during processing.")
    
    # If models have been computed, show the results and prediction section
    if st.session_state.get('models_computed', False):
        # Show model performance metrics if they exist
        if 'df_results' in st.session_state:
            st.subheader("Model Performance Metrics")
            
            # Display all metrics columns
            display_columns = ["Model", "MAE global", "MSE", "RMSE", "R²", "MAE_fast_recipes", "MAE_long_recipes"]
            display_df = st.session_state['df_results'][[col for col in display_columns if col in st.session_state['df_results'].columns]]
            st.dataframe(display_df.style.format(precision=4))
            
        # Recipe prediction section
        st.markdown("---")
        st.subheader("🍳 Predict Cooking Time for a Recipe")
        
        # Get the data from session state
        data = st.session_state['raw_data']
        selected_model_names = st.session_state['selected_models']
        

        def preprocess_duplicate_names(data):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            aggregated_data = data.groupby("name", as_index=False).agg(
                {**{col: "mean" for col in numeric_cols}, "description": "first"}
            )
            return aggregated_data
    
        recipe_names = preprocess_duplicate_names(data)["name"].tolist()
        selected_recipe = st.selectbox(
            "Select a recipe to predict cooking time:",
            options=recipe_names,
            key='recipe_selector'
        )
        
        # Prediction button
        predict_clicked = st.button("Predict Cooking Time", key='predict_button')
        
        if predict_clicked:
            st.session_state['show_predictions'] = True
            st.session_state['selected_recipe'] = selected_recipe
        
        # Show predictions if requested
        if st.session_state.get('show_predictions', False):
            with st.spinner("Generating predictions..."):
                # Get the selected recipe data
                recipe_data = data[data['name'] == st.session_state['selected_recipe']].iloc[0]
                
                # Create a table to display predictions
                prediction_results = []
                
                for model_name in selected_model_names:
                    model_key = models_dict[model_name]
                    model_type = 'tf' if model_key.endswith('tf') else 'w2v'
                    
                    with st.spinner(f"Predicting with {model_name}..."):
                        # Use existing function to predict cooking time
                        predicted_time = predict_cooking_time(
                            model_key,
                            recipe_data,
                            st.session_state['feature_matrices'][model_type],
                            st.session_state['train_test_data'][model_type]
                        )
                        
                        prediction_results.append({
                            "Model": model_name,
                            "Predicted Time (minutes)": predicted_time,
                            "Actual Time (minutes)": recipe_data['minutes']
                        })
                
                pred_df = pd.DataFrame(prediction_results)
                pred_df["Difference (minutes)"] = abs(pred_df["Predicted Time (minutes)"] - pred_df["Actual Time (minutes)"])
                
                # Display recipe info
                st.subheader(f"Recipe: {st.session_state['selected_recipe']}")
                with st.expander("View Recipe Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Recipe Information")
                        st.markdown(f"**Steps Count**: {recipe_data.get('n_steps', 'N/A')}")
                        st.markdown(f"**Ingredients Count**: {recipe_data.get('n_ingredients', 'N/A')}")
                        st.markdown(f"**Actual Cooking Time**: {recipe_data['minutes']} minutes")
                        
                        # Nutrition information if available
                        st.markdown("#### Nutrition Information")
                        nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 'saturated_fat', 'carbohydrates']
                        for col in nutrition_cols:
                            if col in recipe_data and not pd.isna(recipe_data[col]):
                                st.markdown(f"**{col.replace('_', ' ').title()}**: {recipe_data[col]}")
                    
                    with col2:
                        if 'description' in recipe_data and isinstance(recipe_data['description'], str):
                            st.markdown("#### Description")
                            st.markdown(recipe_data['description'][:500] + ('...' if len(recipe_data['description']) > 500 else ''))
                        
                        if 'ingredients_text' in recipe_data and isinstance(recipe_data['ingredients_text'], str):
                            st.markdown("#### Ingredients")
                            
                            # Remove square brackets and single quotes only
                            cleaned_text = recipe_data['ingredients_text'].replace('[', '').replace(']', '').replace("'", "")
                            ingredients = [i.strip() for i in cleaned_text.split(',') if i.strip()]
                            
                            for i in ingredients[:10]:  # Limit to first 10 ingredients
                                st.markdown(f"- {i}")
                            if len(ingredients) > 10:
                                st.markdown(f"*...and {len(ingredients) - 10} more*")
                
                # Display predictions
                st.subheader("Cooking Time Predictions")
                
                # Format the results for better display
                formatted_df = pd.DataFrame(prediction_results)
                formatted_df["Difference (minutes)"] = abs(formatted_df["Predicted Time (minutes)"] - formatted_df["Actual Time (minutes)"])
                formatted_df["Error (%)"] = (formatted_df["Difference (minutes)"] / formatted_df["Actual Time (minutes)"]) * 100
                
                # Style the dataframe
                st.dataframe(
                    formatted_df.style
                    .format({
                        "Predicted Time (minutes)": "{:.1f}", 
                        "Difference (minutes)": "{:.1f}",
                        "Error (%)": "{:.1f}%"
                    })
                    .background_gradient(cmap='RdYlGn_r', subset=['Error (%)'])
                )
                
                # Visualize the predictions
                import plotly.express as px
                
                # Create a copy of the dataframe for plotting
                plot_df = formatted_df.copy()
                
                # Create bar chart for predictions
                fig = px.bar(
                    plot_df, 
                    x="Model", 
                    y="Predicted Time (minutes)",
                    color="Model",
                    title=f"Predicted Cooking Times for {st.session_state['selected_recipe']}",
                    labels={"Predicted Time (minutes)": "Cooking Time (minutes)"}
                )
                
                # Add horizontal line for actual cooking time
                fig.add_hline(
                    y=recipe_data['minutes'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Actual: {recipe_data['minutes']} mins",
                    annotation_position="bottom right"
                )
                
                # Customize the layout
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Cooking Time (minutes)",
                    legend_title="Models",
                    height=450,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add error chart
                fig_error = px.bar(
                    plot_df,
                    x="Model",
                    y="Error (%)",
                    color="Model",
                    title="Prediction Error by Model",
                    labels={"Error (%)": "Error (%)"}
                )
                
                # Customize the error chart
                fig_error.update_layout(
                    xaxis_title="",
                    yaxis_title="Error (%)",
                    legend_title="Models",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
                
                st.plotly_chart(fig_error, use_container_width=True)
        """
        # Add a button to reset predictions and try another recipe
        if st.session_state.get('show_predictions', False):
            if st.button("Try Another Recipe"):
                st.session_state['show_predictions'] = False
        """                
