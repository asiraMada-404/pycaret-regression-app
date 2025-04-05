from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models,get_config,tune_model,pull, create_model,create_docker, predict_model, save_model, finalize_model,load_model,plot_model
from ydata_profiling import ProfileReport
import numpy as np
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import time
import matplotlib.pyplot as plt
import seaborn as sns

def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                        url({image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .copyright {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            color: white;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.5);
            padding: 5px 10px;
            border-radius: 5px;
        }}
        </style>
        <div class="copyright">©️ 2024 Adam Maria-Panagiota | Geoinformatics Engineer </div>
        """,
        unsafe_allow_html=True
    )

# Use the function with your image URL
add_background_image("https://wallpapercat.com/w/full/4/3/5/1210326-3440x1440-desktop-dual-screen-glow-in-the-dark-background.jpg")
    


if os.path.exists('./dataset1.csv'): 
    df = pd.read_csv('dataset1.csv', index_col=None)
if os.path.exists('./prediction_set1.csv'): 
    new_data = pd.read_csv('prediction_set1.csv', index_col=None)
with st.sidebar:
    
    
    st.markdown(
    """
    <div style="opacity: 0.7;">
        <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTFuNW92Y3Vsc2hmOTBnOTc0cWgxOGk2OHJtdnEydGgzNzRlem02MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kE9P0SZ5H3viVCZYct/giphy.webp" alt="Alt Text" />
    </div>
    """,
    unsafe_allow_html=True
    ) 
    #st.image("https://cdn-icons-png.freepik.com/512/8618/8618941.png")
    #st.markdown("![Alt Text](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTFuNW92Y3Vsc2hmOTBnOTc0cWgxOGk2OHJtdnEydGgzNzRlem02MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kE9P0SZ5H3viVCZYct/giphy.webp)")
    st.markdown(
        """
        <div style="text-align: center; font-size: 24px; font-weight: bold;margin-bottom: 10px;">
        Interactive Machine Learning Regression App
        </div>
        """, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; font-size: 16px; font-style: italic; margin-bottom: 40px;">
            Using PyCaret and Streamlit for Real-Time Predictions
        </div>
        """, 
        unsafe_allow_html=True)
    
    # Use markdown to display a bold "Navigation" label
    st.markdown(
        """
        <div style="font-weight: bold; font-size: 16px; margin-bottom: 0px;padding-bottom: 0px;">
            Navigation
        </div>
        """, 
        unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        /* Targeting the radio button component */
        div[data-testid="stRadio"] {
            padding-top: -20px; /* Remove padding on the top */
            margin-top: -40px; /* Adjust margin if necessary */
        }
    
        /* Optional: Set padding and margin to zero for radio labels */
        div[data-testid="stRadio"] > label {
            padding-top: -20px; /* Remove padding from the label */
            margin-top: -20px; /* Remove margin from the label */
        }
        </style>
        """, 
        unsafe_allow_html=True)
    
    # Create the radio button options, with no space between the label and the radio buttons
    choice = st.radio("-",["📤Upload", "📊Profiling","🚀Run Model","⚙️Hyperparameter Tuning","🔮Prediction","📥Download"])

    # Display the image
    #st.image("logo.png", use_column_width=True)
    

    # Add the caption with full alignment using HTML and CSS
    #st.markdown("""
    #    <div style="text-align: left; font-size: 12px; margin-top: 10px;">
    #        ©️2024 Adam Maria-Panagiota | Geoinformatics Eng.
    #    </div>
    #    """, unsafe_allow_html=True)


    
if choice == "📤Upload":
    st.info("In this section you may upload your dataset and then split the data into Data for Modeling & Unseen Data For Predictions")
    st.title("📤Upload Your File")
    file = st.file_uploader("📂Upload Your Dataset")
    
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, index_col=None)
        return df
 
    #if file: 
    #    df = pd.read_csv(file, index_col=None)
    #    df.to_csv('dataset1.csv', index=None) # type: ignore
    #    st.dataframe(df)
    if file: 
        # Load dataset with caching
        df = load_data(file)
        chosen_data_size = st.number_input(
            "👇📝Insert the percentage of records that will be withheld from the original dataset to be used for Modeling",
            min_value=0.0, max_value=1.0, step=0.01, format="%0.2f")
        #st.select_slider('📝Choose the sample of records that will be withhold from the original dataset to be used for Modeling', options=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        # Save for potential use
        df.to_csv('dataset1.csv', index=None)  # type: ignore
        st.write(f"Dataset: {df.shape}") 
        st.dataframe(df)  # Display the full dataset
    
        # Split dataset
        
        
        data = df.sample(frac=chosen_data_size, random_state=786).reset_index(drop=True)
        data_unseen = df.drop(data.index).reset_index(drop=True)
    
        # Store data in session state for later use
        st.session_state['data'] = data
        st.session_state['data_unseen'] = data_unseen
    
        # Display modeling data
        #st.caption('Data for Modeling')
        # Display the shape of data for modeling
        st.write(f"Data for Modeling: {data.shape}") 
        st.dataframe(data)
        data.to_csv('dataformodeling.csv', index=None)  # type: ignore
    
        # Display unseen data for predictions
        #st.caption('Unseen Data For Predictions')
        # Display the shape of unseen data for predictions
        st.write(f"Unseen Data For Predictions: {data_unseen.shape}")  
        st.dataframe(data_unseen)
        data_unseen.to_csv('unseendata.csv', index=None)  # type: ignore

# Check if 'data' is available in session state
if choice == "📊Profiling": 
    if 'data' in st.session_state:
        st.title("📊Exploratory Data Analysis")
        
        # Access data from session state
        data = st.session_state['data']
        
        # Generate profiling report
        profile = ProfileReport(data, title="Data Set Overview", html={'style': {'full_width': True}})
        profile.to_file(output_file="Exploratory Data Analysis.html")
        with open('Exploratory Data Analysis.html', 'rb') as f: 
            st.download_button('Download EDA', f, file_name="Exploratory Data Analysis.html")
        #profile.to_file(output_file="Exploratory Data Analysis.html")
        st_profile_report(profile)
    else:
        st.warning("Dataset is not loaded. Please upload a file first.")
   

if choice == "🚀Run Model":
    if 'data' in st.session_state:
        st.title("🚀PyCaret Regression module")
        # Check if filtered dataset is available in session state
        st.caption("Dataset processing")
        # Access data from session state
        data = st.session_state['data']
        #   Create a multiselect for columns to drop
        columns_to_exclude = st.multiselect("📝Select columns to exclude from dataset", data.columns)
        @st.cache_data
        def process_dataset(data, columns_to_exclude):
            # Simulate some processing
            if columns_to_exclude:
                return data.drop(columns=columns_to_exclude)
            return data.copy()
        df1 = process_dataset(data, columns_to_exclude)
        st.dataframe(df1)
 
        chosen_target = st.selectbox('🎯Choose the Target Column', df1.columns)
        chosen_train_size = st.select_slider('Insert Train Size', options=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]) # type: ignore
        chosen_k_fold = st.select_slider('Insert k-Fold', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # type: ignore
        # Outliers Removal
        remove_outliers_check = st.checkbox("Would you like to remove outliers?")
        outliers_number = None  # Default value if not selected
        if remove_outliers_check:
            outliers_number = st.number_input(
                "👇Insert a number", format="%0.4f", placeholder="Type a number...")
            
            st.write("The chosen outlier threshold is ", outliers_number)
        else:
            st.info("The outliers remain in the dataset.")

        # Normalization
        normalize_check = st.checkbox("Would you like to normalize the data?")
        norm_method_option = None  # Default value if not selected
        if normalize_check:
            norm_method_option = st.selectbox(
                "Which method of normalization would you choose?",
            ("zscore", "minmax", "maxabs", "robust"))
             
            st.write("You selected:", norm_method_option)
        else:
            st.info("No normalization selected.")

        # Multicollinearity Threshold
        chosen_multicollinearity_threshold_check = st.checkbox("Would you like insert a multicollinearity threshold?")
        chosen_multicollinearity_threshold = None  # Default value if not selected
        if chosen_multicollinearity_threshold_check:
            chosen_multicollinearity_threshold = st.number_input(
                "👇Insert a number", format="%0.2f", placeholder="Type a number..."
            )
            st.write("The chosen multicollinearity threshold is ", chosen_multicollinearity_threshold)
        else:
            st.info("No multicollinearity threshold selected.")

        # Data Transformation
        transformation_check = st.checkbox("Would you like to transform the data?")
        transform_method_option = None  # Default value if not selected
        if transformation_check:
            transform_method_option = st.selectbox(
                "Which method of transformation would you choose?",
                ("yeo-johnson", "quantile")
            )
            st.write("You selected:", transform_method_option)
        else:
            st.info("No transformation selected.")
    
    
        if st.button('Run Model'):
        
            st.title("Model Analysis")
            with st.spinner("Processing..."):
         # Setup with progress
                progress_bar = st.progress(0)  # Initialize progress bar
                st.caption("Model Setup")
                #@st.cache_resource
                def cached_model_setup(data, target, train_size, fold, normalize, normalize_method, remove_outliers, outliers_threshold, remove_multicollinearity, multicollinearity_threshold, transformation, transformation_method):

                    return setup(
                        data=df1,
                        target=target,
                        train_size=train_size,
                        fold=fold,
                        normalize=normalize,
                        normalize_method=normalize_method,
                        remove_outliers=remove_outliers,
                        outliers_threshold=outliers_threshold,
                        remove_multicollinearity=remove_multicollinearity,
                        multicollinearity_threshold=multicollinearity_threshold,
                        transformation=transformation,
                        transformation_method=transformation_method
                    )

                setup_df = cached_model_setup(
                    data=df1,
                    target=chosen_target,
                    train_size=chosen_train_size,
                    fold=chosen_k_fold,
                    normalize=normalize_check,
                    normalize_method=norm_method_option if normalize_check else None, # type: ignore
                    remove_outliers=remove_outliers_check,
                    outliers_threshold=outliers_number if remove_outliers_check else None, # type: ignore
                    remove_multicollinearity=chosen_multicollinearity_threshold_check,
                    multicollinearity_threshold=chosen_multicollinearity_threshold if chosen_multicollinearity_threshold_check else None, # type: ignore
                    transformation=transformation_check,
                    transformation_method=transform_method_option if transformation_check else None # type: ignore
                    #session_id=123  # Optional: to ensure reproducibility
                    )  # Optionally normalize the data to handle extreme scales
                progress_bar.progress(10)  # Update progress
                setup_df = pull()
                st.dataframe(setup_df)
                progress_bar.progress(20)  # Update progress
        
                # Retrieve the transformed dataset  
                X_transformed = get_config("X")  # Transformed feature variables
                y_transformed = get_config("y")  # Transformed target variable

                # Combine features and target into a single DataFrame
                transformed_df = X_transformed.copy()
                transformed_df[chosen_target] = y_transformed

            # Print or display the transformed DataFrame
                transform_df = pull()
                st.caption("Transformed Dataset")
                st.dataframe(transformed_df)
                progress_bar.progress(30)

         
            #Compare Models 
                #@st.cache_data
                def cached_compare_models():
           
                    best_model = compare_models()
                    return best_model

                best_model = cached_compare_models()
                compare_df = pull()
                st.caption("Compare Models")
                st.dataframe(compare_df)
                progress_bar.progress(75)  # Update progress
        
                st.caption("Analyze Best Model")
                #@st.cache_data
                def cached_plot_model(_best_model, plot_type):
    
                    plot = plot_model(best_model, plot=plot_type, display_format='streamlit')
                    return plot
                st.caption("Residuals")
                residuals_plot = cached_plot_model(best_model, 'residuals')
                st.caption("Prediction Error")
                error_plot = cached_plot_model(best_model, 'error')
                st.caption("Learning Curve")
                learning_curve_plot = cached_plot_model(best_model, 'learning')
            
                progress_bar.progress(90)  # Update progress
        
                save_model(best_model, 'best_model')
                progress_bar.progress(100)  # Complete progress
                st.success("Model ran successfully - best model saved. You may run the 🔮Prediction")
                st.info("In case there's a need for tuning, you may run the ⚙️Hyperparameter Tuning")
        
        
if choice == "⚙️Hyperparameter Tuning":
    st.title("⚙️Tune Model")
    #st.caption("Load Model")
    
    estimator_check = st.checkbox("Create model")
    estimator_default="rf"
    estimator_option = estimator_default # Default value if not selected
    if estimator_check:
        estimator_option = st.selectbox(
            "Pick a model to your liking",
            ('lr','knn', 'nb', 'dt','svm','rbfsvm','gpc','mlp','ridge','rf','qda','ada','gbc','lda','et','xgboost','lightgbm','catboost')
        )
        st.write("You selected:", estimator_option)
    else:
        st.info("rf is selected.") 
        #-----------------------------------------------------#
        # K-fold
    chosentuned_k_fold = st.select_slider('Insert k-Fold', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # type: ignore
        # N-iter
        #chosentuned_n_iter = st.select_slider('Insert Number of of iterations for the random grid search', options=[ 2, 3, 4, 5, 6, 7, 8, 9, 10]) # type: ignore
    optimize_check = st.checkbox("With which metric would you like to optimize the data?")
    optimize_default= "R2"
    optimize_method_option = optimize_default   # Default value if not selected
    if optimize_check:
            optimize_method_option = st.selectbox(
                    "Select one of the below",
                ("MAE", "MSE","RMSE","R2","MAPE")
            )
            st.write("You selected:", optimize_method_option)
    else:
            st.info("The default metric <R2> is selected.")
        
    library_check = st.checkbox("Which library would you like to use for hyperparameter tuning?")
    library_default="scikit-learn"
    library_method_option = library_default # Default value if not selected
    if library_check:
            library_method_option = st.selectbox(
                 "Select one of the below",
                ("scikit-learn","optuna")
            )
            st.write("You selected:", library_method_option)
    else:
            st.info("The default library <scikit-learn> is selected.")
    
    if st.button('Tune Model'):
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)  # Initialize progress bar
        # Setup with progress
            
            tune_est=create_model(estimator_option if estimator_check else estimator_default)
            tuned_model = tune_model(tune_est,fold=chosentuned_k_fold,optimize = optimize_method_option if optimize_check else optimize_default,search_library = library_method_option if library_check else library_default)
            tuned_df = pull()
            st.caption('Tuned_Model')
            st.dataframe(tuned_df)
            progress_bar.progress(20)
        #-----------------------------------------------------#
            st.caption("Analyze Tuned Model")
        
            st.caption("Residuals of Tuned Model")
            t1=plot_model(tuned_model, plot='residuals',display_format='streamlit')
            progress_bar.progress(40)
            st.caption("Prediction Error of Tuned Model")
            t2=plot_model(tuned_model, plot='error',display_format='streamlit')
            progress_bar.progress(60)
            st.caption("Learning Curve of Tuned Model")
            t3=plot_model(tuned_model, plot='learning',display_format='streamlit')
            progress_bar.progress(80)
            save_model(tuned_model, 'tuned_model')
            progress_bar.progress(100)
            st.success("Model ran successfully - tuned model saved. You may run the 🔮Prediction")
            
# Main app logic
if 'new_data' not in st.session_state:
    st.session_state['new_data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

if choice == "🔮Prediction": 

    st.title("🔮 Run Predictions using the Best or the Tuned Model")
    st.info(
        "💡 Tip: Before running the prediction, ensure that the uploaded file "
        "has the same columns as the dataset used on the 'Run Model' section."
    )

    # # File Uploader
    # pred_file = st.file_uploader("📂 Upload Your Prediction Set")
    # if pred_file:
    #     try:
    #         new_data = pd.read_csv(pred_file)
    #         st.session_state['new_data'] = new_data
    #         st.dataframe(new_data)
    #     except Exception as e:
    #         st.error(f"Error loading file: {e}")

    #     st.caption("Prediction Set Processing")
    #     st.warning(
    #         '⚠️ Drop the same selected columns that were excluded from the Dataset '
    #         '& the Target column on the Prediction set.'
    #     )

    #     # Create a multiselect for columns to drop
    #     columns_to_exclude_2 = st.multiselect("📝 Select columns to exclude from prediction", new_data.columns)

    #     if columns_to_exclude_2:
    #         processed_data = new_data.drop(columns=columns_to_exclude_2)
    #     else:
    #         processed_data = new_data.copy()
    #     st.session_state['processed_data'] = processed_data
    #     st.dataframe(processed_data)
    
        
    #     # Model Selection
    #     model_check = st.checkbox("Choose your preferred model")
    #     model_default = "best_model"
    #     model_option = model_default  # Default value if not selected
    #     if model_check:
    #         model_option = st.selectbox(
    #             "Pick a model to your liking",
    #             ("best_model", "tuned_model")
    #         )
    #         st.write("You selected:", model_option)
    #     else:
    #         st.info("Using the default: best_model.") 

    # # Run Prediction Button
    # if st.button('Run Prediction'):
    #     st.title("Predicted Values")
    #     if st.session_state['processed_data'] is not None:
    #         try:
    #             # Load and finalize model
    #             model = load_model(model_option if model_check else model_default)
    #             # predict on test set
    #             holdout_pred = predict_model(model)
                
    #             final_model = finalize_model(model)

    #             # Predict on processed data
    #             predictions = predict_model(final_model, data=st.session_state['processed_data'])
    #             predictions_dt = pull()
    #             st.caption("Prediction Model Values")


    
    #             st.dataframe(predictions_dt)
        
    #             st.caption("Predictions")
    #             st.dataframe(predictions)

    #             # Save the finalized model
    #             save_model(final_model, 'final_model')
    #             st.success("Predictions completed and model saved.")
                    
    #         except Exception as e:
    #             st.error(f"Error during prediction: {e}")
    #     else:
    #         st.error("No processed data available. Ensure you've uploaded and prepared your dataset.")

    # File Uploader
    pred_file = st.file_uploader("📂 Upload Your Prediction Set")
    if pred_file:
        try:
            new_data = pd.read_csv(pred_file)
            st.session_state['new_data'] = new_data
            st.dataframe(new_data)
        except Exception as e:
            st.error(f"Error loading file: {e}")

        st.caption("Prediction Set Processing")
        st.warning(
            '⚠️ Drop the same selected columns that were excluded from the dataset '
            ' on the prediction set.'
        )

    # Create a multiselect for columns to drop
        columns_to_exclude_2 = st.multiselect("📝 Select columns to exclude from prediction", new_data.columns)

        if columns_to_exclude_2:
            processed_data = new_data.drop(columns=columns_to_exclude_2)
        else:
            processed_data = new_data.copy()

        st.session_state['processed_data'] = processed_data
        st.dataframe(processed_data)

    # Select target column
        chosen_target_prediction = st.selectbox('🎯 Choose the Target Column', new_data.columns)

    # Model Selection
        model_check = st.checkbox("Choose your preferred model")
        model_default = "best_model"
        model_option = model_default  # Default value if not selected
        if model_check:
            model_option = st.selectbox(
                "Pick a model to your liking",
                ("best_model", "tuned_model")
            )
            st.write("You selected:", model_option)
        else:
            st.info("Using the default: best_model.")


    # Run Prediction Button
    if st.button('Run Prediction'):
        st.title("Predicted Values")
    
        if st.session_state['processed_data'] is not None:
            try:
                # Load and finalize model
                model = load_model(model_option if model_check else model_default)
                final_model = finalize_model(model)

                # Predict on processed data
                predictions = predict_model(final_model, data=st.session_state['processed_data'])
                predictions_dt = pull()

                st.caption("Prediction Model Values")
                st.dataframe(predictions_dt)

                st.caption("Predictions")
                st.dataframe(predictions)

                # Save the finalized model
                save_model(final_model, 'final_model')
                st.success("Predictions completed and model saved.")

                # Merge Predictions with Original Uploaded Data
                merged_df = st.session_state['new_data'].copy()
                merged_df["Predicted_Value"] = predictions["prediction_label"]

                # Check if the chosen target exists in the uploaded data
                if chosen_target_prediction in merged_df.columns:
                    actual_values = merged_df[chosen_target_prediction]
                    predicted_values = merged_df["Predicted_Value"]
                
                    # Compute absolute error for coloring
                    merged_df["Error"] = abs(actual_values - predicted_values)

                    # Plot with color based on error size
                    fig, ax = plt.subplots(figsize=(8, 5))
                    scatter = sns.scatterplot(
                        x=actual_values, 
                        y=predicted_values, 
                        hue=merged_df["Error"],  # Color by error
                        palette="coolwarm",  # Gradient from blue (low error) to red (high error)
                        size=merged_df["Error"],  # Point size by error
                        sizes=(20, 200),  # Adjust point size range
                        ax=ax
                    )
                
                    ax.set_xlabel("Actual Target")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Predictions vs Actual Values")
                    # Customize legend
                    legend = plt.legend(
                        title="Error Magnitude",
                        loc="best",        # Fixed position
                        fontsize=10,              # Font size of legend labels
                        title_fontsize=12,        # Font size of legend title
                        frameon=True,             # Add a border around the legend
                        facecolor="white",        # Background color
                        edgecolor="black",        # Border color
                        framealpha=0.8            # Transparency (0 = fully transparent, 1 = solid)
                    )

                    st.pyplot(fig)

                else:
                    st.warning("⚠️ The selected target column is not in the uploaded dataset.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("No processed data available. Ensure you've uploaded and prepared your dataset.")

    
    #predictions.to_csv('final_model.csv')
    #predictions.describe()
    create_docker('my-regression-app-v2')
    
        
if choice == "📥Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Best Model', f, file_name="best_model.pkl")
    with open('tuned_model.pkl', 'rb') as f: 
        st.download_button('Download Tuned Model', f, file_name="tuned_model.pkl")
    with open('final_model.pkl', 'rb') as f: 
        st.download_button('Download Final Model', f, file_name="final_model.pkl")





