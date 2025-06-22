import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import unittest
from sqlalchemy.sql import text
from sqlalchemy import inspect
from tests import common
from foundation_sql.query import SQLTableSchemaDecorator

# --- Test Pydantic Models ---
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class User(BaseModel):
    id: str
    name: str
    email: str
    role: UserRole
    created_at: Optional[datetime] = None

class ProductCategory(str, Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"

class Product(BaseModel):
    id: int
    name: str
    price: float
    category: ProductCategory
    description: Optional[str] = None
    is_active: bool = True

class TestSQLTableSchemaDecorator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class-level test environment."""
        # Create a temporary directory for test templates
        cls.test_templates_dir = Path(tempfile.mkdtemp(prefix='foundation_sql_test_templates_'))
        
        # Copy template files to the test directory
        package_templates_dir = Path(__file__).parent.parent / "foundation_sql" / "templates"
        for template_file in package_templates_dir.glob("*.j2"):
            shutil.copy(template_file, cls.test_templates_dir)
        
        # Set the template directory for tests
        os.environ["FOUNDATION_SQL_TEMPLATE_DIR"] = str(cls.test_templates_dir)
    
    def setUp(self):
        """Set up test environment."""
        # Get environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL")
        self.cache_dir = os.path.join(os.getcwd(), '__sql__')
        
        # Skip tests if API key is not set
        if not self.api_key:
            self.skipTest("Skipping test: OPENAI_API_KEY environment variable must be set")
        
        # Create __sql__ directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the database with a minimal schema
        self.db = common.db.get_db(common.DB_URL)
        self.minimal_schema = """
        CREATE TABLE IF NOT EXISTS test (id INTEGER);
        """
        self.db.init_schema(schema_sql=self.minimal_schema)

    def tearDown(self):
        """Clean up after tests."""
        # Close all database connections
        for _, connection in common.db.DATABASES.items():
            connection.get_engine().dispose()
        common.db.DATABASES.clear()

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test environment."""
        # Clean up test templates directory
        if hasattr(cls, 'test_templates_dir') and cls.test_templates_dir.exists():
            shutil.rmtree(cls.test_templates_dir)

    def test_decorator_usage_pattern(self):
        """Test the decorator usage pattern shown in the example."""
        
        try:
            # Create decorator instance without a schema to force generation from model
            decorator = SQLTableSchemaDecorator(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                cache_dir=self.cache_dir,
                db_url=common.DB_URL,
                regen=True  # Force regeneration of schema
            )
        except Exception as e:
            raise
        
        # Define table function with decorator
        @decorator
        def user_table(user: User) -> str:
            """Generate schema for user table."""
            pass
        
        # Call the function to generate the schema
        schema = user_table()
        
        # Verify the schema was generated
        self.assertIsInstance(schema, str)
        self.assertIn("CREATE TABLE", schema.upper())
        
        # Apply the schema to the database
        self.db.init_schema(schema_sql=schema)
        
        # Verify the table was created in the database
        inspector = inspect(self.db.get_engine())
        
        with self.db.get_engine().connect() as conn:
            # Check if users table exists
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            )
            self.assertIsNotNone(result.fetchone())
            
            # Check if the table has the expected columns
            result = conn.execute(text("PRAGMA table_info(users)"))
            columns = {row[1] for row in result.fetchall()}
            expected_columns = {'id', 'name', 'email', 'role', 'created_at'}
            self.assertTrue(expected_columns.issubset(columns))

    def test_multiple_tables_decorator(self):
        """Test creating multiple tables with the decorator."""
        
        try:
            decorator = SQLTableSchemaDecorator(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                cache_dir=self.cache_dir,
                db_url=common.DB_URL,
                regen=True  # Force regeneration of schema
            )
        except Exception as e:
            raise
        
        # Define user table
        @decorator
        def user_table(user: User) -> str:
            """Generate schema for user table."""
            pass
        
        # Define product table
        @decorator
        def product_table(product: Product) -> str:
            """Generate schema for product table."""
            pass
        
        # Generate and apply schemas
        user_schema = user_table()
        product_schema = product_table()
        
        # Verify schemas were generated
        self.assertIn("CREATE TABLE", user_schema.upper())
        self.assertIn("CREATE TABLE", product_schema.upper())
        
        # Apply the schemas to the database
        self.db.init_schema(schema_sql=user_schema)
        self.db.init_schema(schema_sql=product_schema)
        
        # Verify tables were created in the database
        inspector = inspect(self.db.get_engine())
        
        with self.db.get_engine().connect() as conn:
            # Check users table
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            )
            self.assertIsNotNone(result.fetchone())
            
            # Check products table
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
            )
            self.assertIsNotNone(result.fetchone())