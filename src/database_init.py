"""
Database initialization module for AI-Powered Personal Decision Intelligence System.

This module creates and initializes the SQLite database with the required schema
for storing users, datasets, models, predictions, and recommendations.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime


def get_database_path():
    """Get the path to the database file."""
    # Get the project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    database_dir = project_root / "database"
    database_dir.mkdir(exist_ok=True)
    return database_dir / "user_decisions.db"


def create_database_schema(force_recreate=False):
    """
    Create the database schema with all required tables.
    
    Args:
        force_recreate (bool): If True, delete existing database and recreate.
                              If False (default), use existing database if present.
    
    Tables:
    - users: Store user information
    - datasets: Store dataset metadata
    - models: Store trained model metadata
    - predictions: Store prediction results
    - recommendations: Store AI-generated recommendations
    """
    db_path = get_database_path()
    
    # Remove existing database if force_recreate is True
    if force_recreate and db_path.exists():
        print(f"Removing existing database at {db_path}")
        db_path.unlink()
    elif db_path.exists():
        print(f"Using existing database at {db_path}")
        # Still create tables if they don't exist (safe operation)
    
    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            domain TEXT CHECK(domain IN ('finance', 'career', 'health', 'lifestyle', 'other')),
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    # Create models table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            model_type TEXT NOT NULL CHECK(model_type IN ('classification', 'regression')),
            model_path TEXT NOT NULL,
            metrics TEXT,  -- JSON string storing evaluation metrics
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        )
    """)
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            input_data TEXT,  -- JSON string storing input features
            prediction TEXT NOT NULL,  -- Predicted value/class
            decision_score REAL CHECK(decision_score >= 0 AND decision_score <= 1),
            risk_level TEXT CHECK(risk_level IN ('Low', 'Medium', 'High')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
        )
    """)
    
    # Create recommendations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            recommendation_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_dataset_id ON models(dataset_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_prediction_id ON recommendations(prediction_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print(f"Database schema created successfully at {db_path}")
    return db_path


def initialize_database(force_recreate=False):
    """
    Initialize the database - wrapper function for easy import.
    
    Args:
        force_recreate (bool): If True, delete existing database and recreate.
    """
    return create_database_schema(force_recreate=force_recreate)


if __name__ == "__main__":
    # Run database initialization when script is executed directly
    db_path = create_database_schema()
    print(f"\nDatabase initialized at: {db_path}")
    print("\nSchema created with the following tables:")
    print("  - users")
    print("  - datasets")
    print("  - models")
    print("  - predictions")
    print("  - recommendations")
