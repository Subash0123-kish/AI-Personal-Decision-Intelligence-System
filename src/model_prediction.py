"""
Model Prediction Module
Loads trained model and makes predictions with confidence scores
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Any, Optional, Tuple

# Import feature engineering module
try:
    from .feature_engineering import FeatureEngineer
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.feature_engineering import FeatureEngineer


class ModelPredictor:
    """
    Loads trained models and makes predictions
    Calculates decision scores and risk levels
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model predictor
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.model_type = None
        self.task_type = None
        self.feature_names = None
        self.feature_engineer = FeatureEngineer()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file
        
        Args:
            model_path: Path to saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', 'unknown')
        self.task_type = model_data.get('task_type', 'classification')
        self.feature_names = model_data.get('feature_names', [])
    
    def _preprocess_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw input data to match the model's expected features.
        Handles one-hot encoding and feature alignment automatically.
        
        Args:
            X: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame aligned with model features
        """
        X_processed = X.copy()
        
        # Drop high-cardinality ID-like columns (same logic as training)
        MAX_UNIQUE_FOR_ONEHOT = 50
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        high_card_cols = [col for col in cat_cols if X_processed[col].nunique() > MAX_UNIQUE_FOR_ONEHOT]
        if high_card_cols:
            X_processed = X_processed.drop(columns=high_card_cols)
        
        # One-hot encode remaining categorical columns
        cat_cols_remaining = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols_remaining:
            X_processed = pd.get_dummies(X_processed, columns=cat_cols_remaining, prefix=cat_cols_remaining)
        
        # Align with model's expected features
        if self.feature_names:
            X_processed = X_processed.reindex(columns=self.feature_names, fill_value=0)
        
        return X_processed
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix (raw or preprocessed)
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess if features don't match (raw CSV input)
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                X = self._preprocess_for_prediction(X)
            else:
                X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get prediction probabilities (for classification)
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array or None if not available
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        # Preprocess if features don't match (raw CSV input)
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                X = self._preprocess_for_prediction(X)
            else:
                X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions with confidence scores and risk levels
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions, decision scores, and risk levels
        """
        # Get predictions
        predictions = self.predict(X)
        
        # Get probabilities if available
        probabilities = self.predict_proba(X)
        
        # Calculate decision scores
        if probabilities is not None:
            decision_scores = self.feature_engineer.calculate_decision_score(
                probabilities, task_type='classification'
            )
        else:
            # For regression, normalize predictions
            decision_scores = self.feature_engineer.calculate_decision_score(
                predictions, task_type='regression'
            )
        
        # Assign risk levels
        risk_levels = self.feature_engineer.assign_risk_level(decision_scores)
        
        return {
            'predictions': predictions,
            'decision_scores': decision_scores,
            'risk_levels': risk_levels,
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
    
    def predict_single(self, input_data: Dict[str, Any],
                      feature_order: Optional[list] = None) -> Dict[str, Any]:
        """
        Make prediction for a single data point
        
        Args:
            input_data: Dictionary with feature values
            feature_order: Order of features (if different from model)
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Reorder columns if needed
        if feature_order:
            df = df.reindex(columns=feature_order, fill_value=0)
        elif self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Make prediction
        result = self.predict_with_confidence(df)
        
        return {
            'prediction': result['predictions'][0],
            'decision_score': float(result['decision_scores'][0]),
            'risk_level': result['risk_levels'][0],
            'probabilities': result['probabilities'][0] if result['probabilities'] else None
        }
