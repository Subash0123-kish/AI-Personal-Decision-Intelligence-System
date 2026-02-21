"""
Model Training Module
Trains ML models, evaluates performance, and saves the best model
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Trains and evaluates machine learning models
    Supports both classification and regression tasks
    """
    
    def __init__(self):
        """Initialize model trainer"""
        self.model = None
        self.model_type = None
        self.task_type = None
        self.feature_names = None
        self.metrics = {}
    
    def detect_task_type(self, y: pd.Series) -> str:
        """
        Detect if task is classification or regression
        
        Args:
            y: Target variable
            
        Returns:
            'classification' or 'regression'
        """
        # Check if target is categorical
        if y.dtype == 'object' or y.dtype == 'category':
            return 'classification'
        
        # Check number of unique values
        unique_count = y.nunique()
        total_count = len(y)
        
        # If less than 10% unique values, likely classification
        if unique_count / total_count < 0.1 and unique_count < 20:
            return 'classification'
        
        return 'regression'
    
    def create_model(self, model_name: str = 'random_forest', task_type: str = 'classification'):
        """
        Create a model instance
        
        Args:
            model_name: Model type ('random_forest', 'logistic', 'linear')
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        
        # If auto, defer model creation until train() when we can detect from data
        if task_type == 'auto':
            self.model_type = model_name
            self.task_type = None
            return
        
        if task_type == 'classification':
            if model_name == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            elif model_name == 'logistic':
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # regression
            if model_name == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            elif model_name == 'linear':
                self.model = LinearRegression()
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model_type = model_name
    
    def train(self, X: pd.DataFrame, y: pd.Series,
             test_size: float = 0.2,
             random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        # Detect task type if not set
        if self.task_type is None:
            self.task_type = self.detect_task_type(y)
        
        # Create model if not created
        if self.model is None:
            self.create_model(task_type=self.task_type)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.task_type == 'classification' else None
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_metrics = self.evaluate(X_train, y_train, prefix='train')
        test_metrics = self.evaluate(X_test, y_test, prefix='test')
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, 
                                   scoring='accuracy' if self.task_type == 'classification' else 'r2')
        
        self.metrics = {
            **train_metrics,
            **test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_type': self.model_type,
            'task_type': self.task_type
        }
        
        return self.metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, prefix: str = '') -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True target values
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.model.predict(X)
        
        metrics = {}
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics[f'{prefix}_accuracy'] = accuracy_score(y, y_pred)
            metrics[f'{prefix}_precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            metrics[f'{prefix}_recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            metrics[f'{prefix}_f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # Get prediction probabilities for decision scores
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)
            else:
                y_proba = None
            metrics['y_proba'] = y_proba
        else:
            # Regression metrics
            metrics[f'{prefix}_mse'] = mean_squared_error(y, y_pred)
            metrics[f'{prefix}_rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics[f'{prefix}_mae'] = mean_absolute_error(y, y_pred)
            metrics[f'{prefix}_r2'] = r2_score(y, y_pred)
        
        return metrics
    
    def save_model(self, file_path: str):
        """
        Save trained model to file
        
        Args:
            file_path: Path to save model
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, file_path)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            return dict(enumerate(importances))
        
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance()
        }
