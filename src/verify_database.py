"""
Simple script to verify database schema was created correctly.
"""

import sqlite3
from pathlib import Path
from database_init import get_database_path

def verify_database():
    """Verify that all required tables exist in the database."""
    db_path = get_database_path()
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Run: python src/database_init.py")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    # Expected tables
    expected_tables = ['users', 'datasets', 'models', 'predictions', 'recommendations']
    
    print(f"Database found at: {db_path}")
    print(f"\nTables found: {len(tables)}")
    for table in tables:
        print(f"  [OK] {table}")
    
    # Check if all expected tables exist
    missing_tables = set(expected_tables) - set(tables)
    if missing_tables:
        print(f"\nERROR: Missing tables: {missing_tables}")
        conn.close()
        return False
    
    # Verify indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
    indexes = [row[0] for row in cursor.fetchall()]
    print(f"\nIndexes found: {len(indexes)}")
    for idx in indexes:
        print(f"  [OK] {idx}")
    
    conn.close()
    print("\n[SUCCESS] Database schema verification successful!")
    return True

if __name__ == "__main__":
    verify_database()
