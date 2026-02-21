"""
EDA Utilities Module
Provides functions for Exploratory Data Analysis
Can be used in notebooks and Streamlit dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class EDAUtils:
    """
    Utility functions for Exploratory Data Analysis
    """
    
    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get basic statistical summary
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistics
        """
        return df.describe()
    
    @staticmethod
    def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get missing value information
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value counts and percentages
        """
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_percent.values
        })
        
        return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot correlation matrix
        
        Args:
            df: Input DataFrame
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            print("No numerical columns found for correlation matrix")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        corr_matrix = numerical_df.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_distributions(df: pd.DataFrame, columns: Optional[list] = None, 
                          figsize: Tuple[int, int] = (15, 10)):
        """
        Plot distributions of numerical columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to plot (None = all numerical)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            print("No numerical columns found")
            return None
        
        if columns is None:
            columns = numerical_df.columns.tolist()
        else:
            columns = [col for col in columns if col in numerical_df.columns]
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(columns) > 1 else [axes]
        
        for i, col in enumerate(columns):
            axes[i].hist(numerical_df[col].dropna(), bins=30, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide extra subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_boxplots(df: pd.DataFrame, columns: Optional[list] = None,
                     figsize: Tuple[int, int] = (15, 10)):
        """
        Plot boxplots for numerical columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            print("No numerical columns found")
            return None
        
        if columns is None:
            columns = numerical_df.columns.tolist()
        else:
            columns = [col for col in columns if col in numerical_df.columns]
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(columns) > 1 else [axes]
        
        for i, col in enumerate(columns):
            axes[i].boxplot(numerical_df[col].dropna())
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_ylabel(col)
        
        # Hide extra subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_target_analysis(df: pd.DataFrame, target_column: str):
        """
        Analyze target variable distribution
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Matplotlib figure
        """
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        target_data = df[target_column]
        
        # Distribution plot
        if target_data.dtype == 'object' or target_data.nunique() < 20:
            # Categorical target
            value_counts = target_data.value_counts().head(20)
            axes[0].bar(value_counts.index.astype(str), value_counts.values)
            title_suffix = ' (Top 20)' if target_data.nunique() > 20 else ''
            axes[0].set_title(f'Distribution of {target_column}{title_suffix}')
            axes[0].set_xlabel(target_column)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
        else:
            # Numerical target
            axes[0].hist(target_data.dropna(), bins=30, edgecolor='black')
            axes[0].set_title(f'Distribution of {target_column}')
            axes[0].set_xlabel(target_column)
            axes[0].set_ylabel('Frequency')
        
        # Statistics
        missing_count = target_data.isnull().sum()
        missing_pct = missing_count / len(target_data) * 100 if len(target_data) > 0 else 0
        
        if pd.api.types.is_numeric_dtype(target_data):
            stats_text = f"""
        Statistics for {target_column}:
        Mean: {target_data.mean():.2f}
        Median: {target_data.median():.2f}
        Std: {target_data.std():.2f}
        Min: {target_data.min():.2f}
        Max: {target_data.max():.2f}
        Missing: {missing_count} ({missing_pct:.1f}%)
            """
        else:
            top_value = target_data.mode().iloc[0] if not target_data.mode().empty else 'N/A'
            stats_text = f"""
        Statistics for {target_column}:
        Unique Values: {target_data.nunique()}
        Most Common: {top_value}
        Total Records: {len(target_data)}
        Missing: {missing_count} ({missing_pct:.1f}%)
            """
        
        axes[1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_eda_report(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
        """
        Generate comprehensive EDA report
        
        Args:
            df: Input DataFrame
            target_column: Optional target column name
            
        Returns:
            Dictionary with EDA results
        """
        report = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'basic_stats': EDAUtils.get_basic_stats(df).to_dict(),
            'missing_info': EDAUtils.get_missing_info(df).to_dict('records'),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        if target_column and target_column in df.columns:
            target_col = df[target_column]
            if pd.api.types.is_numeric_dtype(target_col):
                report['target_stats'] = {
                    'mean': float(target_col.mean()),
                    'median': float(target_col.median()),
                    'std': float(target_col.std()),
                    'min': float(target_col.min()),
                    'max': float(target_col.max()),
                    'unique_count': int(target_col.nunique())
                }
            else:
                report['target_stats'] = {
                    'unique_count': int(target_col.nunique()),
                    'most_common': str(target_col.mode().iloc[0]) if not target_col.mode().empty else 'N/A',
                    'total_records': len(target_col),
                    'missing_count': int(target_col.isnull().sum())
                }
        
        return report
