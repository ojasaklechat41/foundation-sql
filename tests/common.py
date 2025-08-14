import unittest
import os
import tempfile
from foundation_sql import db
from foundation_sql.query import SQLQueryDecorator

from dotenv import load_dotenv
load_dotenv()

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///:memory:")  # Fixed typo: DATABSE_URL -> DATABASE_URL

def create_query(schema):
    # Use a unique temp cache dir per invocation to prevent cross-test contamination
    cache_dir = tempfile.mkdtemp(prefix='foundation_sql_test_cache_')
    return SQLQueryDecorator(
        schema=schema,
        db_url=DB_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        cache_dir=cache_dir,
        regen=True,  # force fresh generation for each test
    )

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
        # If a schema is provided, initialize it; otherwise just ensure DB instance exists.
        try:
            db_instance = db.get_db(self.db_url)
            if self.schema_sql or self.schema_path:
                db_instance.init_schema(schema_sql=self.schema_sql, schema_path=self.schema_path)
        except Exception as e:
            self.fail(f"Failed to initialize database or schema: {e}")

    def tearDown(self):
        """Close the database connection after each test."""
        for _, connection in db.DATABASES.items():
            try:
                connection.get_engine().dispose()
            except:
                pass  # Ignore errors when disposing connections
        
        db.DATABASES.clear()