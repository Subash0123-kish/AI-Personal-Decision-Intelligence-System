"""
Streamlit Dashboard for AI-Powered Personal Decision Intelligence System
Main user interface for data upload, model training, predictions, and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_prediction import ModelPredictor
from src.explainability import ModelExplainer
from src.ai_recommendation import AIRecommender
from src.database_manager import DatabaseManager
from src.eda_utils import EDAUtils

# Page configuration
st.set_page_config(
    page_title="AI Decision Intelligence System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium background and styling
import base64

def get_bg_image():
    bg_path = os.path.join(os.path.dirname(__file__), "assets", "bg.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"
    return None

bg_url = get_bg_image()
bg_css = f"""
    background-image: url("{bg_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
""" if bg_url else "background: #0a0e27;"

st.markdown(f"""
<style>
/* Background image */
.stApp {{
    {bg_css}
}}

/* Dark overlay for readability */
.stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(5, 5, 20, 0.75);
    pointer-events: none;
    z-index: 0;
}}

/* Glowing accent line at top */
.stApp::after {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #6495ED, #8A2BE2, #00CED1, #6495ED);
    background-size: 200% 100%;
    animation: glowLine 3s linear infinite;
    z-index: 999;
}}

@keyframes glowLine {{
    0% {{ background-position: 0% 0%; }}
    100% {{ background-position: 200% 0%; }}
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background: rgba(10, 14, 39, 0.92) !important;
    border-right: 1px solid rgba(100, 149, 237, 0.15);
    backdrop-filter: blur(15px);
}}

/* Metric cards glow */
[data-testid="stMetric"] {{
    background: rgba(26, 16, 64, 0.5);
    border: 1px solid rgba(138, 43, 226, 0.2);
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.1);
}}

/* Button styling */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #6495ED, #8A2BE2) !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(100, 149, 237, 0.3);
    transition: all 0.3s ease;
}}

.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 30px rgba(100, 149, 237, 0.5);
    transform: translateY(-2px);
}}

/* Dataframe styling */
[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(100, 149, 237, 0.15);
}}

/* Headers glow effect */
h1 {{
    text-shadow: 0 0 30px rgba(100, 149, 237, 0.3);
}}

h2 {{
    text-shadow: 0 0 20px rgba(138, 43, 226, 0.2);
}}

/* Main content area */
.stMainBlockContainer {{
    position: relative;
    z-index: 1;
}}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None

# Initialize components
db_manager = DatabaseManager()
data_processor = DataProcessor()
feature_engineer = FeatureEngineer()
eda_utils = EDAUtils()

# Sidebar navigation
st.sidebar.title("🤖 Decision Intelligence")
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Data Upload", "EDA Dashboard", "Model Training", "Predictions", "Explanations", "Recommendations", "History"]
)

# Home Page
if page == "Home":
    st.title("AI-Powered Personal Decision Intelligence System")
    st.markdown("""
    Welcome! This system helps you make informed decisions using AI and machine learning.
    
    ### Features:
    - 📊 **Data Analysis**: Upload your data and explore it with interactive visualizations
    - 🤖 **ML Models**: Train models to predict outcomes
    - 🔍 **Explainability**: Understand WHY the model made its predictions
    - 💡 **AI Recommendations**: Get actionable advice based on predictions
    - 📈 **Decision Scores**: Quantify decision risk levels
    
    ### Getting Started:
    1. **Data Upload**: Upload your CSV file or enter data manually
    2. **EDA Dashboard**: Explore your data with visualizations
    3. **Model Training**: Train a model on your data
    4. **Predictions**: Make predictions on new data
    5. **Explanations**: Understand model decisions
    6. **Recommendations**: Get AI-powered advice
    """)

# Data Upload Page
elif page == "Data Upload":
    st.title("📤 Data Upload")
    
    upload_method = st.radio("Choose input method:", ["Upload CSV", "Manual Entry"])
    
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"Data loaded successfully! Shape: {df.shape}")
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                st.subheader("Data Info")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Rows:** {df.shape[0]}")
                    st.write(f"**Columns:** {df.shape[1]}")
                with col2:
                    st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # Select target column
                st.subheader("Select Target Column")
                all_columns = df.columns.tolist()
                target_col = st.selectbox("Target column (for prediction):", ["None"] + all_columns)
                if target_col != "None":
                    st.session_state.target_column = target_col
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    else:  # Manual Entry
        st.subheader("Enter Data Manually")
        num_rows = st.number_input("Number of rows:", min_value=1, max_value=100, value=5)
        
        # Get column names
        col_names_input = st.text_input("Column names (comma-separated):", "feature1, feature2, target")
        col_names = [col.strip() for col in col_names_input.split(",")]
        
        # Create form for data entry
        data_dict = {}
        for col in col_names:
            values = st.text_input(f"Values for {col} (comma-separated):", "")
            if values:
                data_dict[col] = [v.strip() for v in values.split(",")]
        
        if data_dict:
            try:
                df = pd.DataFrame(data_dict)
                st.session_state.data = df
                st.success("Data entered successfully!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error creating DataFrame: {e}")

# EDA Dashboard
elif page == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload' page.")
    else:
        df = st.session_state.data
        
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Basic Statistics
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Missing Values
        missing_info = eda_utils.get_missing_info(df)
        if not missing_info.empty:
            st.subheader("Missing Values")
            st.dataframe(missing_info)
        
        # Visualizations
        st.subheader("Visualizations")
        
        viz_type = st.selectbox("Select visualization:", 
                               ["Correlation Matrix", "Distributions", "Boxplots", "Target Analysis"])
        
        if viz_type == "Correlation Matrix":
            fig = eda_utils.plot_correlation_matrix(df)
            if fig:
                st.pyplot(fig)
        
        elif viz_type == "Distributions":
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                selected_cols = st.multiselect("Select columns:", numerical_cols, default=numerical_cols[:3])
                if selected_cols:
                    fig = eda_utils.plot_distributions(df, columns=selected_cols)
                    if fig:
                        st.pyplot(fig)
            else:
                st.info("No numerical columns found for distribution plots.")
        
        elif viz_type == "Boxplots":
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                selected_cols = st.multiselect("Select columns:", numerical_cols, default=numerical_cols[:3])
                if selected_cols:
                    fig = eda_utils.plot_boxplots(df, columns=selected_cols)
                    if fig:
                        st.pyplot(fig)
            else:
                st.info("No numerical columns found for boxplots.")
        
        elif viz_type == "Target Analysis":
            if st.session_state.target_column:
                fig = eda_utils.plot_target_analysis(df, st.session_state.target_column)
                if fig:
                    st.pyplot(fig)
            else:
                st.info("Please select a target column in the Data Upload page.")

# Model Training Page
elif page == "Model Training":
    st.title("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
    elif st.session_state.target_column is None:
        st.warning("Please select a target column in the Data Upload page.")
    else:
        df = st.session_state.data
        target_col = st.session_state.target_column
        
        st.subheader("Training Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type:", ["random_forest", "logistic", "linear"])
        with col2:
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
        
        col3, col4 = st.columns(2)
        with col3:
            missing_strategy = st.selectbox("Missing Value Strategy:", ["mean", "median", "mode", "drop"])
        with col4:
            scaler_type = st.selectbox("Scaler Type:", ["standard", "minmax"])
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Preprocess data
                    X_processed, y_processed = data_processor.preprocess(
                        df, target_column=target_col,
                        missing_strategy=missing_strategy,
                        scaler_type=scaler_type,
                        fit=True
                    )
                    
                    st.session_state.processed_data = X_processed
                    
                    # Train model
                    trainer = ModelTrainer()
                    trainer.create_model(model_type, task_type='auto')
                    metrics = trainer.train(X_processed, y_processed, test_size=test_size)
                    
                    # Save model
                    model_path = "models/decision_model.pkl"
                    trainer.save_model(model_path)
                    st.session_state.model = trainer
                    st.session_state.model_path = model_path
                    
                    # Save model to database for history tracking
                    try:
                        user_id = db_manager.create_user("default_user")
                        dataset_id = db_manager.create_dataset(
                            user_id=user_id, name="uploaded_data", 
                            domain="general", file_path="uploaded"
                        )
                        # Convert numpy types to native Python for JSON serialization
                        serializable_metrics = {}
                        for k, v in metrics.items():
                            try:
                                serializable_metrics[k] = float(v) if isinstance(v, (int, float)) or hasattr(v, 'item') else str(v)
                            except (TypeError, ValueError):
                                serializable_metrics[k] = str(v)
                        db_manager.save_model(
                            dataset_id=dataset_id, model_type=model_type,
                            model_path=model_path, metrics=serializable_metrics
                        )
                    except Exception as db_err:
                        st.warning(f"Model trained but could not save to DB for history: {db_err}")
                    
                    # Create predictor
                    predictor = ModelPredictor(model_path)
                    st.session_state.predictor = predictor
                    
                    st.success("Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    metrics_cols = st.columns(len(metrics) // 2 + 1)
                    for i, (key, value) in enumerate(metrics.items()):
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            with metrics_cols[i % len(metrics_cols)]:
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                    
                    # Feature Importance
                    feature_importance = trainer.get_feature_importance()
                    if feature_importance:
                        st.subheader("Top Feature Importance")
                        importance_df = pd.DataFrame(
                            list(feature_importance.items())[:10],
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature', 
                                    orientation='h', title='Top 10 Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    st.exception(e)

# Predictions Page
elif page == "Predictions":
    st.title("🔮 Make Predictions")
    
    if st.session_state.predictor is None:
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        predictor = st.session_state.predictor
        
        st.subheader("Input Data for Prediction")
        input_method = st.radio("Input method:", ["Manual Entry", "Upload CSV"])
        
        if input_method == "Manual Entry":
            # Get feature names from model
            if predictor.feature_names:
                input_data = {}
                for feature in predictor.feature_names[:10]:  # Limit to first 10 for UI
                    value = st.number_input(f"{feature}:", value=0.0)
                    input_data[feature] = value
                
                # Fill remaining features with 0
                for feature in predictor.feature_names[10:]:
                    input_data[feature] = 0.0
                
                if st.button("Predict", type="primary"):
                    try:
                        result = predictor.predict_single(input_data)
                        
                        st.subheader("Prediction Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction", f"{result['prediction']:.4f}")
                        with col2:
                            st.metric("Decision Score", f"{result['decision_score']:.2%}")
                        with col3:
                            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                            st.metric("Risk Level", f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}")
                        
                        # Store for explanations
                        st.session_state.last_prediction = result
                        st.session_state.last_input_data = input_data
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
        
        else:  # Upload CSV
            uploaded_file = st.file_uploader("Upload CSV with features", type=['csv'])
            if uploaded_file is not None:
                try:
                    pred_df = pd.read_csv(uploaded_file)
                    # Drop target column if present
                    if st.session_state.target_column and st.session_state.target_column in pred_df.columns:
                        pred_df = pred_df.drop(columns=[st.session_state.target_column])
                    # Preprocess the data the same way the predictor will
                    pred_processed = predictor._preprocess_for_prediction(pred_df)
                    results = predictor.predict_with_confidence(pred_df)
                    
                    # Save to session state for Explanations page
                    st.session_state.last_prediction = {
                        'prediction': results['predictions'][0],
                        'decision_score': float(results['decision_scores'][0]),
                        'risk_level': results['risk_levels'][0],
                        'probabilities': results['probabilities'][0] if results['probabilities'] else None
                    }
                    st.session_state.last_input_data = pred_processed.iloc[0].to_dict()
                    st.session_state.last_pred_df = pred_processed
                    
                    st.subheader("Predictions")
                    results_df = pd.DataFrame({
                        'Prediction': results['predictions'],
                        'Decision Score': results['decision_scores'],
                        'Risk Level': results['risk_levels']
                    })
                    st.dataframe(results_df)
                    
                    # Visualization
                    fig = px.scatter(results_df, x='Decision Score', y='Prediction', 
                                    color='Risk Level', title='Predictions Overview')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save predictions to database for History page
                    try:
                        if st.session_state.model_path:
                            latest_model = db_manager.get_latest_model()
                            if latest_model:
                                model_id = latest_model['id']
                                # Save first 100 predictions to avoid DB overload
                                for i in range(min(len(results['predictions']), 100)):
                                    db_manager.save_prediction(
                                        model_id=model_id,
                                        input_data={'row_index': i},
                                        prediction=float(results['predictions'][i]),
                                        decision_score=float(results['decision_scores'][i]),
                                        risk_level=str(results['risk_levels'][i])
                                    )
                                st.success(f"✅ {min(len(results['predictions']), 100)} predictions saved to history!")
                    except Exception as db_err:
                        st.warning(f"Predictions shown but could not save to history: {db_err}")
                
                except Exception as e:
                    st.error(f"Error processing predictions: {e}")

# Explanations Page
elif page == "Explanations":
    st.title("🔍 Model Explanations")
    
    if st.session_state.model is None or st.session_state.processed_data is None:
        st.warning("Please train a model first.")
    elif 'last_prediction' not in st.session_state or st.session_state.last_prediction is None:
        st.warning("Please make a prediction first in the 'Predictions' page.")
    else:
        model = st.session_state.model.model
        X_sample = st.session_state.processed_data.sample(min(100, len(st.session_state.processed_data)))
        
        try:
            explainer = ModelExplainer(model, X_sample)
            st.session_state.explainer = explainer
            
            # Get local explanation for the first prediction
            if 'last_input_data' in st.session_state and st.session_state.last_input_data:
                last_input_df = pd.DataFrame([st.session_state.last_input_data])
                # Align columns with training data
                last_input_df = last_input_df.reindex(columns=X_sample.columns, fill_value=0)
            elif 'last_pred_df' in st.session_state:
                last_input_df = st.session_state.last_pred_df.head(1)
                last_input_df = last_input_df.reindex(columns=X_sample.columns, fill_value=0)
            else:
                last_input_df = X_sample.head(1)
            
            local_explanation = explainer.explain_local(last_input_df)
            
            st.subheader("Feature Contributions")
            contributions_df = pd.DataFrame(
                list(local_explanation['feature_contributions'].items())[:10],
                columns=['Feature', 'Contribution']
            )
            
            fig = px.bar(contributions_df, x='Contribution', y='Feature', 
                        orientation='h', title='Top Feature Contributions (SHAP Values)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation text
            explanation_text = explainer.get_explanation_text(last_input_df)
            st.subheader("Explanation")
            st.text(explanation_text)
            
            # Global explanation
            st.subheader("Global Feature Importance")
            global_explanation = explainer.explain_global()
            importance_df = pd.DataFrame(
                list(global_explanation['feature_importance'].items())[:10],
                columns=['Feature', 'Importance']
            )
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title='Global Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating explanations: {e}")
            st.exception(e)

# Recommendations Page
elif page == "Recommendations":
    st.title("💡 AI Recommendations")
    
    if 'last_prediction' not in st.session_state:
        st.warning("Please make a prediction first.")
    else:
        prediction_result = st.session_state.last_prediction
        
        st.subheader("Prediction Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Decision Score", f"{prediction_result['decision_score']:.2%}")
        with col2:
            st.metric("Risk Level", prediction_result['risk_level'])
        with col3:
            st.metric("Prediction", f"{prediction_result['prediction']:.4f}")
        
        domain = st.selectbox("Domain Context:", ["finance", "career", "health", "lifestyle", "general"])
        
        if st.button("Generate Recommendation", type="primary"):
            with st.spinner("Generating AI recommendation..."):
                try:
                    # Get feature contributions if available
                    feature_contributions = {}
                    feature_values = {}
                    if st.session_state.explainer and 'last_input_data' in st.session_state:
                        local_expl = st.session_state.explainer.explain_local(
                            pd.DataFrame([st.session_state.last_input_data])
                        )
                        feature_contributions = local_expl['feature_contributions']
                        feature_values = local_expl['feature_values']
                    
                    # Generate recommendation
                    recommender = AIRecommender()
                    recommendation = recommender.generate_recommendation(
                        prediction=prediction_result['prediction'],
                        decision_score=prediction_result['decision_score'],
                        risk_level=prediction_result['risk_level'],
                        domain=domain,
                        feature_contributions=feature_contributions,
                        feature_values=feature_values
                    )
                    
                    st.subheader("AI-Generated Recommendation")
                    st.markdown(f"**{domain.title()} Decision Advice:**")
                    st.info(recommendation)
                    
                    # Save to database
                    if st.session_state.model_path:
                        latest_model = db_manager.get_latest_model()
                        if latest_model:
                            model_id = latest_model['id']
                            pred_id = db_manager.save_prediction(
                                model_id=model_id,
                                input_data=st.session_state.last_input_data,
                                prediction=prediction_result['prediction'],
                                decision_score=prediction_result['decision_score'],
                                risk_level=prediction_result['risk_level']
                            )
                            db_manager.save_recommendation(pred_id, recommendation)
                            st.success("Recommendation saved to history!")
                
                except Exception as e:
                    st.error(f"Error generating recommendation: {e}")
                    st.exception(e)

# History Page
elif page == "History":
    st.title("📜 Prediction History")
    
    history = db_manager.get_predictions_history(limit=20)
    
    if history:
        st.subheader("Recent Predictions")
        for record in history:
            with st.expander(f"Prediction #{record['id']} - {record['created_at']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Decision Score:** {record['decision_score']:.2%}")
                with col2:
                    st.write(f"**Risk Level:** {record['risk_level']}")
                with col3:
                    st.write(f"**Prediction:** {record['prediction']:.4f}")
                
                if record.get('recommendation_text'):
                    st.write("**Recommendation:**")
                    st.info(record['recommendation_text'])
    else:
        st.info("No prediction history yet. Make some predictions to see them here!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**AI Decision Intelligence System**")
st.sidebar.markdown("Built with Streamlit, Scikit-learn, and SHAP")
