"""
Feature Engineering Module
Creates decision scores, risk indicators, and aggregated metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class FeatureEngineer:
    """
    Creates engineered features for decision intelligence:
    - Decision scores
    - Risk indicators
    - Aggregated metrics
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def calculate_decision_score(self, predictions: np.ndarray, 
                                task_type: str = 'classification') -> np.ndarray:
        """
        Calculate decision scores from model predictions
        
        Args:
            predictions: Model predictions (probabilities for classification, raw values for regression)
            task_type: 'classification' or 'regression'
            
        Returns:
            Decision scores normalized to 0-1 scale
        """
        if task_type == 'classification':
            # For classification, use probability as decision score
            if predictions.ndim > 1:
                # Multi-class: use max probability
                decision_scores = np.max(predictions, axis=1)
            else:
                # Binary: use probability of positive class
                decision_scores = predictions
        else:
            # For regression, normalize to 0-1 scale
            min_val = np.min(predictions)
            max_val = np.max(predictions)
            if max_val - min_val > 0:
                decision_scores = (predictions - min_val) / (max_val - min_val)
            else:
                decision_scores = np.ones_like(predictions) * 0.5
        
        return decision_scores
    
    def assign_risk_level(self, decision_scores: np.ndarray) -> List[str]:
        """
        Assign risk levels based on decision scores
        
        Args:
            decision_scores: Decision scores (0-1 scale)
            
        Returns:
            List of risk levels ('Low', 'Medium', 'High')
        """
        risk_levels = []
        for score in decision_scores:
            if score < 0.33:
                risk_levels.append('Low')
            elif score < 0.67:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        return risk_levels
    
    def create_risk_indicators(self, df: pd.DataFrame, 
                              risk_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create risk indicator features from existing columns
        
        Args:
            df: Input DataFrame
            risk_features: List of feature names to use for risk calculation
            
        Returns:
            DataFrame with risk indicators added
        """
        df_risk = df.copy()
        
        if risk_features is None:
            # Auto-detect numerical columns
            risk_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate risk indicators
        for feature in risk_features:
            if feature in df.columns:
                # Normalize feature
                feature_min = df[feature].min()
                feature_max = df[feature].max()
                if feature_max - feature_min > 0:
                    normalized = (df[feature] - feature_min) / (feature_max - feature_min)
                    df_risk[f'{feature}_risk'] = normalized
        
        return df_risk
    
    def create_aggregated_metrics(self, df: pd.DataFrame,
                                 group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Create aggregated metrics (mean, std, min, max) for features
        
        Args:
            df: Input DataFrame
            group_by: Column name to group by (optional)
            
        Returns:
            DataFrame with aggregated metrics
        """
        df_agg = df.copy()
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if group_by and group_by in df.columns:
            # Grouped aggregation
            agg_dict = {}
            for col in numerical_cols:
                agg_dict[col] = ['mean', 'std', 'min', 'max']
            
            grouped = df.groupby(group_by).agg(agg_dict)
            grouped.columns = [f'{col}_{stat}' for col, stat in grouped.columns]
            df_agg = pd.merge(df_agg, grouped, left_on=group_by, right_index=True, how='left')
        else:
            # Overall aggregation
            for col in numerical_cols:
                df_agg[f'{col}_mean'] = df[col].mean()
                df_agg[f'{col}_std'] = df[col].std()
                df_agg[f'{col}_min'] = df[col].min()
                df_agg[f'{col}_max'] = df[col].max()
        
        return df_agg
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Create interaction features (multiplication, division, etc.)
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples to create interactions
            
        Returns:
            DataFrame with interaction features
        """
        df_inter = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if feature_pairs is None:
            # Auto-create interactions for top correlated pairs
            if len(numerical_cols) >= 2:
                corr_matrix = df[numerical_cols].corr().abs()
                # Get top pairs (excluding self-correlations)
                corr_matrix = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                top_pairs = corr_matrix.stack().nlargest(min(5, len(numerical_cols) * (len(numerical_cols) - 1) // 2))
                feature_pairs = [(pair[0], pair[1]) for pair in top_pairs.index]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                df_inter[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                # Division (avoid division by zero)
                df_inter[f'{feat1}_div_{feat2}'] = np.where(
                    df[feat2] != 0,
                    df[feat1] / df[feat2],
                    0
                )
        
        return df_inter
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                   degree: int = 2,
                                   features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            features: List of features to create polynomials for (None = all numerical)
            
        Returns:
            DataFrame with polynomial features
        """
        df_poly = df.copy()
        
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    df_poly[f'{feature}^{d}'] = df[feature] ** d
        
        return df_poly
    
    def engineer_features(self, df: pd.DataFrame,
                         include_risk_indicators: bool = True,
                         include_aggregated: bool = False,
                         include_interactions: bool = False,
                         include_polynomials: bool = False) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            include_risk_indicators: Whether to add risk indicators
            include_aggregated: Whether to add aggregated metrics
            include_interactions: Whether to add interaction features
            include_polynomials: Whether to add polynomial features
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        if include_risk_indicators:
            df_engineered = self.create_risk_indicators(df_engineered)
        
        if include_aggregated:
            df_engineered = self.create_aggregated_metrics(df_engineered)
        
        if include_interactions:
            df_engineered = self.create_interaction_features(df_engineered)
        
        if include_polynomials:
            df_engineered = self.create_polynomial_features(df_engineered)
        
        return df_engineered
