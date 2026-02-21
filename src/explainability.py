"""
Explainability Module
Uses SHAP to explain model predictions and feature importance
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Explains model predictions using SHAP values
    Provides both local (per-prediction) and global (model-level) explanations
    """
    
    def __init__(self, model: Any, X: pd.DataFrame):
        """
        Initialize explainer
        
        Args:
            model: Trained ML model
            X: Feature matrix (training data or sample)
        """
        self.model = model
        self.X = X
        self.explainer = None
        self.shap_values = None
        self._create_explainer()
    
    def _create_explainer(self):
        """Create SHAP explainer based on model type"""
        try:
            # Try TreeExplainer first (for tree-based models)
            if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models (slower but more general)
                # Use a sample of data for faster computation
                sample_size = min(100, len(self.X))
                X_sample = self.X.sample(n=sample_size, random_state=42)
                self.explainer = shap.KernelExplainer(self.model.predict, X_sample)
        except Exception as e:
            # Fallback to KernelExplainer
            print(f"Warning: Could not create TreeExplainer, using KernelExplainer: {e}")
            sample_size = min(100, len(self.X))
            X_sample = self.X.sample(n=sample_size, random_state=42)
            self.explainer = shap.KernelExplainer(self.model.predict, X_sample)
    
    def explain_global(self, X_explain: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate global explanations (model behavior)
        
        Args:
            X_explain: Data to explain (uses self.X if None)
            
        Returns:
            Dictionary with SHAP values and feature importance
        """
        if X_explain is None:
            X_explain = self.X
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_explain)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # For multi-class, use the first class or average
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        except:
            # Fallback: calculate for a sample
            sample_size = min(50, len(X_explain))
            X_sample = X_explain.sample(n=sample_size, random_state=42)
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        self.shap_values = shap_values
        
        # Calculate feature importance (mean absolute SHAP values)
        if isinstance(shap_values, np.ndarray):
            feature_importance = np.abs(shap_values).mean(axis=0)
        else:
            feature_importance = np.abs(shap_values).mean()
        
        # Create feature importance dictionary
        feature_names = X_explain.columns.tolist()
        importance_dict = dict(zip(feature_names, feature_importance))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_dict,
            'feature_names': feature_names
        }
    
    def explain_local(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction
        
        Args:
            X_instance: Single instance to explain
            
        Returns:
            Dictionary with SHAP values and feature contributions
        """
        # Calculate SHAP values for this instance
        try:
            shap_values = self.explainer.shap_values(X_instance)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        except:
            # Fallback
            shap_values = self.explainer.shap_values(X_instance.iloc[0:1])
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        # Get feature values
        feature_values = X_instance.iloc[0].to_dict()
        feature_names = X_instance.columns.tolist()
        
        # Create contribution dictionary
        if isinstance(shap_values, np.ndarray):
            contributions = shap_values[0] if shap_values.ndim > 1 else shap_values
        else:
            contributions = shap_values
        
        contributions_dict = dict(zip(feature_names, contributions))
        
        # Sort by absolute contribution
        contributions_dict = dict(sorted(
            contributions_dict.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return {
            'shap_values': contributions,
            'feature_contributions': contributions_dict,
            'feature_values': feature_values,
            'feature_names': feature_names
        }
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features and their importance scores
        """
        global_explanation = self.explain_global()
        importance = global_explanation['feature_importance']
        
        # Return top N
        return dict(list(importance.items())[:top_n])
    
    def plot_summary(self, X_explain: Optional[pd.DataFrame] = None, 
                    max_display: int = 10, show: bool = False):
        """
        Create SHAP summary plot
        
        Args:
            X_explain: Data to explain
            max_display: Maximum number of features to display
            show: Whether to display plot immediately
        """
        if X_explain is None:
            X_explain = self.X
        
        global_explanation = self.explain_global(X_explain)
        shap_values = global_explanation['shap_values']
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, max_display=max_display, show=show)
        
        return plt.gcf()
    
    def plot_waterfall(self, X_instance: pd.DataFrame, show: bool = False):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X_instance: Single instance to explain
            show: Whether to display plot immediately
            
        Returns:
            Matplotlib figure
        """
        local_explanation = self.explain_local(X_instance)
        shap_values = local_explanation['shap_values']
        
        # Create waterfall plot
        try:
            # Try to use Explainer object for waterfall
            if hasattr(shap, 'waterfall_plot'):
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values,
                        base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                        data=X_instance.iloc[0].values,
                        feature_names=X_instance.columns.tolist()
                    ),
                    show=show
                )
            else:
                # Fallback: bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_names = local_explanation['feature_names']
                contributions = shap_values if isinstance(shap_values, np.ndarray) else np.array([shap_values])
                
                # Sort by absolute value
                indices = np.argsort(np.abs(contributions))[::-1]
                sorted_names = [feature_names[i] for i in indices]
                sorted_values = contributions[indices]
                
                ax.barh(sorted_names, sorted_values)
                ax.set_xlabel('SHAP Value')
                ax.set_title('Feature Contributions to Prediction')
                plt.tight_layout()
                
                if show:
                    plt.show()
                
                return fig
        except Exception as e:
            print(f"Could not create waterfall plot: {e}")
            # Simple bar plot fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            contributions_dict = local_explanation['feature_contributions']
            features = list(contributions_dict.keys())[:10]
            values = list(contributions_dict.values())[:10]
            
            ax.barh(features, values)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Top Feature Contributions')
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return fig
    
    def get_explanation_text(self, X_instance: pd.DataFrame, top_n: int = 5) -> str:
        """
        Generate human-readable explanation text
        
        Args:
            X_instance: Single instance to explain
            top_n: Number of top features to include
            
        Returns:
            Explanation text
        """
        local_explanation = self.explain_local(X_instance)
        contributions = local_explanation['feature_contributions']
        feature_values = local_explanation['feature_values']
        
        # Get top contributing features
        top_features = list(contributions.items())[:top_n]
        
        explanation_parts = []
        explanation_parts.append("Model Prediction Explanation:\n")
        explanation_parts.append("Top contributing factors:\n")
        
        for i, (feature, contribution) in enumerate(top_features, 1):
            direction = "increases" if contribution > 0 else "decreases"
            value = feature_values.get(feature, "N/A")
            explanation_parts.append(
                f"{i}. {feature} (value: {value:.2f}) {direction} the prediction by {abs(contribution):.4f}"
            )
        
        return "\n".join(explanation_parts)
