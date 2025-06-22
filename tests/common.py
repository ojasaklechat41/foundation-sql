import unittest
import os
from foundation_sql import db
from foundation_sql.query import SQLQueryDecorator

from dotenv import load_dotenv
load_dotenv()

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///:memory:")  # Fixed typo: DATABSE_URL -> DATABASE_URL

def create_query(schema):
    return SQLQueryDecorator(schema=schema, 
                   db_url=DB_URL, 
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("OPENAI_BASE_URL"),
                   model=os.getenv("OPENAI_MODEL"))

class DatabaseTests(unittest.TestCase):
    """Base test class for database-driven tests with common setup and helper methods."""

    db_url = DB_URL
    schema_sql = None
    schema_path = None

    def setUp(self):
        """Create a fresh database connection for each test."""
        # Clear any existing connections first to ensure clean state
        for _, connection in db.DATABASES.items():
            try:
                connection.get_engine().dispose()
            except:
                pass  # Ignore errors when disposing connections
        
        db.DATABASES.clear()
        
        # Re-initialize the schema for each test to ensure clean state
        if (self.schema_sql or self.schema_path) and self.db_url:
            try:
                db_instance = db.get_db(self.db_url)
                db_instance.init_schema(schema_sql=self.schema_sql, schema_path=self.schema_path)
            except Exception as e:
                self.fail(f"Failed to initialize database schema: {e}")
        else:
            raise ValueError("At least one of schema_sql, schema_path must be provided along with db_url")

    def tearDown(self):
        """Close the database connection after each test."""
        for _, connection in db.DATABASES.items():
            try:
                connection.get_engine().dispose()
            except:
                pass  # Ignore errors when disposing connections
        
        db.DATABASES.clear()