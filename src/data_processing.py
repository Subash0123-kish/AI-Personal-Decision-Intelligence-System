"""
Data Processing Module
Handles data loading, cleaning, and preprocessing for ML models
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handles all data processing operations:
    - Loading CSV files or form data
    - Handling missing values
    - Encoding categorical variables
    - Scaling numerical features
    - Outlier detection and handling
    """
    
    def __init__(self):
        """Initialize data processor"""
        self.scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.imputer = None
        self.target_encoder = None
        self.feature_names = None
        self.categorical_columns = []
        self.numerical_columns = []
    
    def load_data(self, file_path: Optional[str] = None, 
                  data: Optional[pd.DataFrame] = None,
                  form_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Load data from CSV file, DataFrame, or form input
        
        Args:
            file_path: Path to CSV file
            data: Existing pandas DataFrame
            form_data: Dictionary of form inputs
            
        Returns:
            Loaded DataFrame
        """
        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path)
        elif data is not None:
            df = data.copy()
        elif form_data:
            # Convert form data dictionary to DataFrame
            df = pd.DataFrame([form_data])
        else:
            raise ValueError("Must provide file_path, data, or form_data")
        
        return df
    
    def detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Automatically detect categorical and numerical columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        categorical = []
        numerical = []
        
        for col in df.columns:
            # Skip target column if it exists (will be handled separately)
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                # Check if it's actually numeric but stored as string
                try:
                    pd.to_numeric(df[col].dropna())
                    numerical.append(col)
                except:
                    categorical.append(col)
            else:
                numerical.append(col)
        
        self.categorical_columns = categorical
        self.numerical_columns = numerical
        return categorical, numerical
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Separate categorical and numerical columns
        cat_cols, num_cols = self.detect_data_types(df_clean)
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        else:
            # Handle numerical columns
            if num_cols:
                if strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                else:
                    imputer = SimpleImputer(strategy='mean')
                
                # Ensure numeric columns are actually numeric dtype
                for col in num_cols:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Filter to only columns that have at least one non-NaN value
                valid_num_cols = [c for c in num_cols if df_clean[c].notna().any()]
                
                if valid_num_cols:
                    imputed_values = imputer.fit_transform(df_clean[valid_num_cols])
                    df_clean[valid_num_cols] = pd.DataFrame(
                        imputed_values, columns=valid_num_cols, index=df_clean.index
                    )
                self.imputer = imputer
            
            # Handle categorical columns
            if cat_cols:
                for col in cat_cols:
                    if df_clean[col].isna().any():
                        if strategy == 'mode':
                            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
                        else:
                            df_clean[col].fillna('Unknown', inplace=True)
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, 
                          encoding_type: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            encoding_type: 'onehot' or 'label'
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        cat_cols, _ = self.detect_data_types(df_encoded)
        
        if not cat_cols:
            return df_encoded
        
        # Separate high-cardinality columns (likely IDs) from encodable ones
        MAX_UNIQUE_FOR_ONEHOT = 50
        high_cardinality_cols = [col for col in cat_cols 
                                 if df_encoded[col].nunique() > MAX_UNIQUE_FOR_ONEHOT]
        low_cardinality_cols = [col for col in cat_cols 
                                if df_encoded[col].nunique() <= MAX_UNIQUE_FOR_ONEHOT]
        
        # Drop high-cardinality columns (they're likely IDs, not useful features)
        if high_cardinality_cols:
            df_encoded = df_encoded.drop(columns=high_cardinality_cols)
        
        if encoding_type == 'onehot':
            # One-hot encoding only low-cardinality columns
            if low_cardinality_cols:
                df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality_cols, prefix=low_cardinality_cols)
        elif encoding_type == 'label':
            # Label encoding
            for col in low_cardinality_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, 
                      scaler_type: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            scaler_type: 'standard' or 'minmax'
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        _, num_cols = self.detect_data_types(df_scaled)
        
        if not num_cols:
            return df_scaled
        
        if scaler_type == 'standard':
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                df_scaled[num_cols] = pd.DataFrame(
                    self.scaler.fit_transform(df_scaled[num_cols]),
                    columns=num_cols, index=df_scaled.index
                )
            else:
                df_scaled[num_cols] = pd.DataFrame(
                    self.scaler.transform(df_scaled[num_cols]),
                    columns=num_cols, index=df_scaled.index
                )
        elif scaler_type == 'minmax':
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler()
                df_scaled[num_cols] = pd.DataFrame(
                    self.scaler.fit_transform(df_scaled[num_cols]),
                    columns=num_cols, index=df_scaled.index
                )
            else:
                df_scaled[num_cols] = pd.DataFrame(
                    self.scaler.transform(df_scaled[num_cols]),
                    columns=num_cols, index=df_scaled.index
                )
        
        return df_scaled
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers using IQR method
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr')
            
        Returns:
            DataFrame with outlier information
        """
        df_outliers = df.copy()
        _, num_cols = self.detect_data_types(df_outliers)
        
        outlier_mask = pd.Series([False] * len(df_outliers))
        
        if method == 'iqr':
            for col in num_cols:
                Q1 = df_outliers[col].quantile(0.25)
                Q3 = df_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = (df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        return outlier_mask
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from dataset
        
        Args:
            df: Input DataFrame
            method: Outlier detection method
            
        Returns:
            DataFrame with outliers removed
        """
        outlier_mask = self.detect_outliers(df, method)
        return df[~outlier_mask]
    
    def preprocess(self, df: pd.DataFrame, 
                  target_column: Optional[str] = None,
                  missing_strategy: str = 'mean',
                  encoding_type: str = 'onehot',
                  scaler_type: str = 'standard',
                  remove_outliers: bool = False,
                  fit: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if exists)
            missing_strategy: Strategy for handling missing values
            encoding_type: Categorical encoding type
            scaler_type: Feature scaling type
            remove_outliers: Whether to remove outliers
            fit: Whether to fit transformers (True for training data)
            
        Returns:
            Tuple of (processed_features, target_series)
        """
        # Separate target if specified
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column].copy()
            df = df.drop(columns=[target_column])
            
            # Label-encode the target if it contains non-numeric values
            if target.dtype == 'object' or target.dtype == 'category':
                self.target_encoder = LabelEncoder()
                target = pd.Series(
                    self.target_encoder.fit_transform(target.astype(str)),
                    index=target.index,
                    name=target_column
                )
        
        # Step 1: Handle missing values
        df_clean = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Step 2: Remove outliers if requested
        if remove_outliers:
            df_clean = self.remove_outliers(df_clean)
            if target is not None:
                outlier_mask = self.detect_outliers(df, method='iqr')
                target = target[~outlier_mask]
        
        # Step 3: Encode categorical variables
        df_encoded = self.encode_categorical(df_clean, encoding_type=encoding_type)
        
        # Step 4: Scale features
        df_scaled = self.scale_features(df_encoded, scaler_type=scaler_type, fit=fit)
        
        # Store feature names
        self.feature_names = df_scaled.columns.tolist()
        
        return df_scaled, target
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str):
        """
        Save processed data to CSV
        
        Args:
            df: Processed DataFrame
            file_path: Output file path
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    
    def get_feature_names(self) -> list:
        """Get list of feature names after preprocessing"""
        return self.feature_names if self.feature_names else []
