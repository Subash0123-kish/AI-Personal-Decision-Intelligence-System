"""
Exploratory Data Analysis (EDA) Module for AI-Powered Personal Decision Intelligence System.

This module provides comprehensive statistical analysis and visualization functions:
- Correlation matrices
- Distribution plots (histograms, box plots)
- Missing value analysis
- Feature statistics (mean, median, std, etc.)
- Target variable analysis
- Interactive visualizations using Plotly
- Export EDA reports as images/HTML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
from io import BytesIO
import base64

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for matplotlib/seaborn plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDAAnalyzer:
    """
    Main class for Exploratory Data Analysis operations.
    
    Provides comprehensive statistical analysis and visualization capabilities
    for understanding datasets before machine learning modeling.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize the EDA Analyzer.
        
        Args:
            df: Input DataFrame to analyze
            target_column: Optional name of target column (for supervised analysis)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.target = None
        
        # Separate target if specified
        if target_column and target_column in self.df.columns:
            self.target = self.df[target_column].copy()
            self.features_df = self.df.drop(columns=[target_column])
        else:
            self.features_df = self.df
        
        # Detect data types
        self.data_types = self._detect_data_types()
        
        # Storage for analysis results
        self.statistics: Dict[str, Any] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.missing_value_summary: Optional[pd.DataFrame] = None
        
    def _detect_data_types(self) -> Dict[str, str]:
        """
        Detect data types for each column.
        
        Returns:
            Dictionary mapping column names to types ('numeric', 'categorical', 'datetime', 'boolean')
        """
        type_mapping = {}
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].dtype == bool or self.df[col].nunique() == 2:
                    type_mapping[col] = 'boolean'
                else:
                    type_mapping[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                type_mapping[col] = 'datetime'
            else:
                type_mapping[col] = 'categorical'
        
        return type_mapping
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """
        Calculate basic statistical summary for all columns.
        
        Returns:
            DataFrame with statistics (count, mean, std, min, 25%, 50%, 75%, max)
        """
        stats = self.df.describe(include='all').T
        
        # Add additional statistics
        stats['missing_count'] = self.df.isnull().sum()
        stats['missing_percentage'] = (stats['missing_count'] / len(self.df)) * 100
        stats['unique_count'] = [self.df[col].nunique() for col in stats.index]
        stats['dtype'] = [str(self.df[col].dtype) for col in stats.index]
        
        self.statistics['basic'] = stats
        return stats
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Returns:
            DataFrame with missing value summary (column, missing_count, missing_percentage)
        """
        missing_data = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': self.df.isnull().sum().values,
            'missing_percentage': (self.df.isnull().sum().values / len(self.df)) * 100,
            'data_type': [self.data_types.get(col, 'unknown') for col in self.df.columns]
        })
        
        missing_data = missing_data.sort_values('missing_percentage', ascending=False)
        missing_data = missing_data[missing_data['missing_count'] > 0]  # Only show columns with missing values
        
        self.missing_value_summary = missing_data
        return missing_data
    
    def plot_missing_values(self, figsize: Tuple[int, int] = (10, 6)) -> go.Figure:
        """
        Create interactive visualization of missing values.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Plotly figure object
        """
        missing_data = self.analyze_missing_values()
        
        if len(missing_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found in the dataset!",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Missing Values Analysis")
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=missing_data['column'],
            y=missing_data['missing_percentage'],
            name='Missing Percentage',
            marker_color='coral',
            text=[f"{p:.1f}%" for p in missing_data['missing_percentage']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Missing Values Analysis",
            xaxis_title="Column",
            yaxis_title="Missing Percentage (%)",
            xaxis={'categoryorder': 'total descending'},
            height=figsize[1] * 80,
            width=figsize[0] * 80
        )
        
        return fig
    
    def calculate_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        numeric_cols = [col for col, dtype in self.data_types.items() if dtype == 'numeric']
        
        if len(numeric_cols) < 2:
            print("Warning: Need at least 2 numeric columns for correlation analysis")
            return pd.DataFrame()
        
        corr_matrix = self.df[numeric_cols].corr(method=method)
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def plot_correlation_matrix(self, method: str = 'pearson', figsize: Tuple[int, int] = (12, 10)) -> go.Figure:
        """
        Create interactive correlation matrix heatmap.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size (width, height)
            
        Returns:
            Plotly figure object
        """
        corr_matrix = self.calculate_correlation_matrix(method=method)
        
        if corr_matrix.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough numeric columns for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=f"Correlation Matrix ({method.title()})")
            return fig
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=f"Correlation Matrix ({method.title()})",
            xaxis_title="Features",
            yaxis_title="Features",
            height=figsize[1] * 80,
            width=figsize[0] * 80
        )
        
        return fig
    
    def plot_distribution(
        self, 
        column: str, 
        bins: int = 30, 
        figsize: Tuple[int, int] = (10, 6)
    ) -> go.Figure:
        """
        Plot distribution of a numeric column (histogram + box plot).
        
        Args:
            column: Name of column to plot
            bins: Number of bins for histogram
            figsize: Figure size (width, height)
            
        Returns:
            Plotly figure object with subplots
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if self.data_types.get(column) != 'numeric':
            raise ValueError(f"Column '{column}' is not numeric. Use plot_categorical_distribution() instead.")
        
        # Create subplots: histogram on top, box plot on bottom
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Distribution of {column}', f'Box Plot of {column}'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=self.df[column],
                nbinsx=bins,
                name='Frequency',
                marker_color='skyblue',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=self.df[column],
                name=column,
                marker_color='lightcoral',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add statistics annotations
        mean_val = self.df[column].mean()
        median_val = self.df[column].median()
        std_val = self.df[column].std()
        
        fig.add_annotation(
            text=f"Mean: {mean_val:.2f} | Median: {median_val:.2f} | Std: {std_val:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=1.02,
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.update_xaxes(title_text=column, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text=column, row=2, col=1)
        
        fig.update_layout(
            title=f"Distribution Analysis: {column}",
            height=figsize[1] * 80,
            width=figsize[0] * 80,
            showlegend=False
        )
        
        return fig
    
    def plot_categorical_distribution(
        self, 
        column: str, 
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> go.Figure:
        """
        Plot distribution of a categorical column (bar chart).
        
        Args:
            column: Name of column to plot
            top_n: Show only top N categories (None for all)
            figsize: Figure size (width, height)
            
        Returns:
            Plotly figure object
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        value_counts = self.df[column].value_counts()
        
        if top_n:
            value_counts = value_counts.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            name='Count',
            marker_color='steelblue',
            text=value_counts.values,
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            xaxis={'categoryorder': 'total descending'},
            height=figsize[1] * 80,
            width=figsize[0] * 80
        )
        
        return fig
    
    def plot_all_distributions(
        self, 
        numeric_bins: int = 30,
        categorical_top_n: int = 10,
        max_columns: int = 6
    ) -> List[go.Figure]:
        """
        Plot distributions for all columns in the dataset.
        
        Args:
            numeric_bins: Number of bins for numeric histograms
            categorical_top_n: Top N categories to show for categorical columns
            max_columns: Maximum number of columns per row in subplot grid
            
        Returns:
            List of Plotly figure objects
        """
        figures = []
        
        numeric_cols = [col for col, dtype in self.data_types.items() if dtype == 'numeric']
        categorical_cols = [col for col, dtype in self.data_types.items() if dtype == 'categorical']
        
        # Plot numeric distributions
        for col in numeric_cols:
            fig = self.plot_distribution(col, bins=numeric_bins)
            figures.append(fig)
        
        # Plot categorical distributions
        for col in categorical_cols:
            fig = self.plot_categorical_distribution(col, top_n=categorical_top_n)
            figures.append(fig)
        
        return figures
    
    def analyze_target_variable(self) -> Dict[str, Any]:
        """
        Analyze target variable (if specified).
        
        Returns:
            Dictionary with target variable analysis results
        """
        if self.target is None:
            return {"error": "No target variable specified"}
        
        analysis = {}
        
        # Basic statistics
        if self.data_types.get(self.target_column) == 'numeric':
            analysis['type'] = 'regression'
            analysis['statistics'] = {
                'count': len(self.target),
                'mean': float(self.target.mean()),
                'median': float(self.target.median()),
                'std': float(self.target.std()),
                'min': float(self.target.min()),
                'max': float(self.target.max()),
                'skewness': float(self.target.skew()),
                'kurtosis': float(self.target.kurtosis())
            }
        else:
            analysis['type'] = 'classification'
            value_counts = self.target.value_counts()
            analysis['statistics'] = {
                'count': len(self.target),
                'unique_classes': int(self.target.nunique()),
                'class_distribution': value_counts.to_dict(),
                'class_percentages': (value_counts / len(self.target) * 100).to_dict(),
                'is_balanced': bool((value_counts / len(self.target)).min() > 0.1)  # At least 10% in each class
            }
        
        return analysis
    
    def plot_target_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> go.Figure:
        """
        Plot distribution of target variable.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Plotly figure object
        """
        if self.target is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No target variable specified",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Target Variable Distribution")
            return fig
        
        if self.data_types.get(self.target_column) == 'numeric':
            # Regression target: histogram + box plot
            fig = self.plot_distribution(self.target_column, figsize=figsize)
        else:
            # Classification target: bar chart
            fig = self.plot_categorical_distribution(self.target_column, figsize=figsize)
        
        return fig
    
    def plot_feature_vs_target(
        self, 
        feature_column: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> go.Figure:
        """
        Plot relationship between a feature and target variable.
        
        Args:
            feature_column: Name of feature column
            figsize: Figure size (width, height)
            
        Returns:
            Plotly figure object
        """
        if self.target is None:
            raise ValueError("No target variable specified")
        
        if feature_column not in self.features_df.columns:
            raise ValueError(f"Feature '{feature_column}' not found")
        
        feature_type = self.data_types.get(feature_column)
        target_type = self.data_types.get(self.target_column)
        
        # Numeric feature vs Numeric target: Scatter plot
        if feature_type == 'numeric' and target_type == 'numeric':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.features_df[feature_column],
                y=self.target,
                mode='markers',
                marker=dict(
                    color='steelblue',
                    size=5,
                    opacity=0.6
                ),
                name='Data Points'
            ))
            
            # Add trend line
            z = np.polyfit(self.features_df[feature_column].dropna(), 
                          self.target[self.features_df[feature_column].dropna().index], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=self.features_df[feature_column].sort_values(),
                y=p(self.features_df[feature_column].sort_values()),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"{feature_column} vs {self.target_column}",
                xaxis_title=feature_column,
                yaxis_title=self.target_column,
                height=figsize[1] * 80,
                width=figsize[0] * 80
            )
        
        # Numeric feature vs Categorical target: Box plots
        elif feature_type == 'numeric' and target_type == 'categorical':
            fig = go.Figure()
            
            for category in self.target.unique():
                fig.add_trace(go.Box(
                    y=self.features_df[self.target == category][feature_column],
                    name=str(category),
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f"{feature_column} by {self.target_column}",
                xaxis_title=self.target_column,
                yaxis_title=feature_column,
                height=figsize[1] * 80,
                width=figsize[0] * 80
            )
        
        # Categorical feature vs Numeric target: Box plots
        elif feature_type == 'categorical' and target_type == 'numeric':
            fig = go.Figure()
            
            for category in self.features_df[feature_column].unique():
                fig.add_trace(go.Box(
                    y=self.target[self.features_df[feature_column] == category],
                    name=str(category),
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f"{self.target_column} by {feature_column}",
                xaxis_title=feature_column,
                yaxis_title=self.target_column,
                height=figsize[1] * 80,
                width=figsize[0] * 80
            )
        
        # Categorical feature vs Categorical target: Stacked bar chart
        else:
            crosstab = pd.crosstab(self.features_df[feature_column], self.target)
            fig = go.Figure()
            
            for category in crosstab.columns:
                fig.add_trace(go.Bar(
                    x=crosstab.index.astype(str),
                    y=crosstab[category],
                    name=str(category)
                ))
            
            fig.update_layout(
                title=f"{feature_column} vs {self.target_column}",
                xaxis_title=feature_column,
                yaxis_title="Count",
                barmode='stack',
                height=figsize[1] * 80,
                width=figsize[0] * 80
            )
        
        return fig
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the EDA.
        
        Returns:
            String containing the summary report
        """
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset Overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 80)
        report.append(f"Total Rows: {len(self.df):,}")
        report.append(f"Total Columns: {len(self.df.columns)}")
        report.append(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")
        
        # Data Types
        report.append("DATA TYPES")
        report.append("-" * 80)
        type_counts = {}
        for dtype in self.data_types.values():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        for dtype, count in type_counts.items():
            report.append(f"  {dtype.capitalize()}: {count}")
        report.append("")
        
        # Missing Values
        missing_data = self.analyze_missing_values()
        report.append("MISSING VALUES")
        report.append("-" * 80)
        if len(missing_data) == 0:
            report.append("  No missing values found!")
        else:
            report.append(f"  Columns with missing values: {len(missing_data)}")
            for _, row in missing_data.head(10).iterrows():
                report.append(f"    {row['column']}: {row['missing_count']} ({row['missing_percentage']:.2f}%)")
        report.append("")
        
        # Basic Statistics
        stats = self.get_basic_statistics()
        report.append("NUMERIC COLUMNS SUMMARY")
        report.append("-" * 80)
        numeric_cols = [col for col, dtype in self.data_types.items() if dtype == 'numeric']
        if numeric_cols:
            for col in numeric_cols[:5]:  # Show first 5
                report.append(f"  {col}:")
                report.append(f"    Mean: {stats.loc[col, 'mean']:.2f}")
                report.append(f"    Std: {stats.loc[col, 'std']:.2f}")
                report.append(f"    Min: {stats.loc[col, 'min']:.2f}")
                report.append(f"    Max: {stats.loc[col, 'max']:.2f}")
        else:
            report.append("  No numeric columns found")
        report.append("")
        
        # Target Variable Analysis
        if self.target is not None:
            target_analysis = self.analyze_target_variable()
            report.append("TARGET VARIABLE ANALYSIS")
            report.append("-" * 80)
            report.append(f"  Target Column: {self.target_column}")
            report.append(f"  Type: {target_analysis.get('type', 'unknown')}")
            if 'statistics' in target_analysis:
                for key, value in target_analysis['statistics'].items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for k, v in value.items():
                            report.append(f"    {k}: {v}")
                    else:
                        report.append(f"  {key}: {value}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_report_html(
        self, 
        output_path: Optional[Union[str, Path]] = None,
        include_plots: bool = True
    ) -> Path:
        """
        Export EDA report as HTML file.
        
        Args:
            output_path: Path to save HTML file. If None, saves to project_root/reports/eda_report.html
            include_plots: Whether to include interactive plots in HTML
            
        Returns:
            Path to saved HTML file
        """
        if output_path is None:
            project_root = Path(__file__).parent.parent
            reports_dir = project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            output_path = reports_dir / "eda_report.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = []
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html><head>")
        html_content.append("<title>EDA Report</title>")
        html_content.append("<meta charset='utf-8'>")
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_content.append("h1, h2 { color: #333; }")
        html_content.append("pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }")
        html_content.append("</style>")
        html_content.append("</head><body>")
        
        # Add summary report
        html_content.append("<h1>Exploratory Data Analysis Report</h1>")
        html_content.append("<pre>")
        html_content.append(self.generate_summary_report())
        html_content.append("</pre>")
        
        # Add plots if requested
        if include_plots:
            html_content.append("<h2>Visualizations</h2>")
            
            # Missing values plot
            missing_fig = self.plot_missing_values()
            html_content.append("<h3>Missing Values</h3>")
            html_content.append(missing_fig.to_html(include_plotlyjs='cdn', div_id="missing_values"))
            
            # Correlation matrix
            corr_fig = self.plot_correlation_matrix()
            html_content.append("<h3>Correlation Matrix</h3>")
            html_content.append(corr_fig.to_html(include_plotlyjs='cdn', div_id="correlation"))
            
            # Target distribution if available
            if self.target is not None:
                target_fig = self.plot_target_distribution()
                html_content.append("<h3>Target Variable Distribution</h3>")
                html_content.append(target_fig.to_html(include_plotlyjs='cdn', div_id="target_dist"))
        
        html_content.append("</body></html>")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        
        print(f"EDA report exported to: {output_path}")
        return output_path


# Convenience functions for easy usage
def analyze_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    export_html: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> EDAAnalyzer:
    """
    Convenience function to perform complete EDA on a dataset.
    
    Args:
        df: Input DataFrame
        target_column: Optional name of target column
        export_html: Whether to export HTML report
        output_path: Path for HTML export (if export_html=True)
        
    Returns:
        EDAAnalyzer instance with all analysis results
    """
    analyzer = EDAAnalyzer(df, target_column=target_column)
    
    # Perform all analyses
    analyzer.get_basic_statistics()
    analyzer.analyze_missing_values()
    analyzer.calculate_correlation_matrix()
    
    if target_column:
        analyzer.analyze_target_variable()
    
    # Export HTML if requested
    if export_html:
        analyzer.export_report_html(output_path=output_path)
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("EDA Module - Example Usage")
    print("=" * 60)
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = {
        'age': np.random.randint(20, 60, 100),
        'income': np.random.normal(50000, 15000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.normal(75, 10, 100),
        'target': np.random.choice([0, 1], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create analyzer
    analyzer = EDAAnalyzer(df, target_column='target')
    
    # Generate summary
    print("\n" + analyzer.generate_summary_report())
    
    # Create some plots
    print("\nGenerating visualizations...")
    missing_fig = analyzer.plot_missing_values()
    corr_fig = analyzer.plot_correlation_matrix()
    target_fig = analyzer.plot_target_distribution()
    
    print("\nEDA analysis complete!")
