"""
Database Manager for SQLite operations
Handles all database interactions for storing users, datasets, predictions, and recommendations
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import json


class DatabaseManager:
    """
    Manages SQLite database operations for the Decision Intelligence System
    """
    
    def __init__(self, db_path: str = "database/user_decisions.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _get_connection(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return self.conn
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                domain TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                input_data TEXT,
                prediction REAL,
                decision_score REAL,
                risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        
        # Recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                recommendation_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
        
        conn.commit()
    
    def create_user(self, name: str, email: Optional[str] = None) -> int:
        """
        Create a new user
        
        Args:
            name: User name
            email: User email (optional)
            
        Returns:
            User ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        conn.commit()
        return cursor.lastrowid
    
    def create_dataset(self, user_id: int, name: str, domain: str, file_path: str) -> int:
        """
        Create a new dataset record
        
        Args:
            user_id: User ID
            name: Dataset name
            domain: Domain (finance, career, health, lifestyle)
            file_path: Path to dataset file
            
        Returns:
            Dataset ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO datasets (user_id, name, domain, file_path) VALUES (?, ?, ?, ?)",
            (user_id, name, domain, file_path)
        )
        conn.commit()
        return cursor.lastrowid
    
    def save_model(self, dataset_id: int, model_type: str, model_path: str, metrics: Dict[str, Any]) -> int:
        """
        Save model information
        
        Args:
            dataset_id: Dataset ID
            model_type: Type of model (RandomForest, XGBoost, etc.)
            model_path: Path to saved model file
            metrics: Model evaluation metrics
            
        Returns:
            Model ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO models (dataset_id, model_type, model_path, metrics) VALUES (?, ?, ?, ?)",
            (dataset_id, model_type, model_path, json.dumps(metrics))
        )
        conn.commit()
        return cursor.lastrowid
    
    def save_prediction(self, model_id: int, input_data: Dict[str, Any], 
                       prediction: float, decision_score: float, risk_level: str) -> int:
        """
        Save a prediction
        
        Args:
            model_id: Model ID
            input_data: Input features as dictionary
            prediction: Model prediction
            decision_score: Decision score (0-1)
            risk_level: Risk level (Low/Medium/High)
            
        Returns:
            Prediction ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO predictions (model_id, input_data, prediction, decision_score, risk_level) 
               VALUES (?, ?, ?, ?, ?)""",
            (model_id, json.dumps(input_data), prediction, decision_score, risk_level)
        )
        conn.commit()
        return cursor.lastrowid
    
    def save_recommendation(self, prediction_id: int, recommendation_text: str) -> int:
        """
        Save a recommendation
        
        Args:
            prediction_id: Prediction ID
            recommendation_text: AI-generated recommendation text
            
        Returns:
            Recommendation ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO recommendations (prediction_id, recommendation_text) VALUES (?, ?)",
            (prediction_id, recommendation_text)
        )
        conn.commit()
        return cursor.lastrowid
    
    def get_predictions_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get prediction history
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of prediction records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*, m.model_type, r.recommendation_text
            FROM predictions p
            LEFT JOIN models m ON p.model_id = m.id
            LEFT JOIN recommendations r ON r.prediction_id = p.id
            ORDER BY p.created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently trained model
        
        Returns:
            Model record or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM models
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
