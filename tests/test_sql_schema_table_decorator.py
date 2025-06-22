import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from sqlalchemy.sql import text

from foundation_sql.query import SQLTableSchemaDecorator
from tests import common

# --- Test Enums ---
class UserRole(str, Enum):
    """User role enumeration for testing."""
    ADMIN = "admin"
    USER = "user"    
    GUEST = "guest"

class ProductCategory(str, Enum):
    """Product category enumeration for testing."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"

# --- Test Models ---
class TestUser(BaseModel):
    """Test user model for schema generation."""
    id: str
    name: str
    email: str
    role: UserRole
    created_at: Optional[datetime] = None

class TestProduct(BaseModel):
    """Test product model for schema generation."""
    id: int
    name: str
    price: float
    category: ProductCategory
    description: Optional[str] = None
    is_active: bool = True

class TestSQLTableSchemaDecorator(common.DatabaseTests):
    """Test cases for SQLTableSchemaDecorator with class-based decoration."""
    
    # Define test schema
    schema_sql = """
    CREATE TABLE IF NOT EXISTS test (id INTEGER);
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test environment."""
        super().setUpClass()
        
        # Create a temporary directory for test templates
        cls.test_templates_dir = Path(tempfile.mkdtemp(prefix='foundation_sql_test_templates_'))
        
        # Copy template files to the test directory
        package_templates_dir = Path(__file__).parent.parent / "foundation_sql" / "templates"
        for template_file in package_templates_dir.glob("*.j2"):
            shutil.copy(template_file, cls.test_templates_dir)
        
        # Set the template directory for tests
        os.environ["FOUNDATION_SQL_TEMPLATE_DIR"] = str(cls.test_templates_dir)
        
        # Initialize the decorator
        cls.schema_decorator = SQLTableSchemaDecorator(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            cache_dir='__sql__',
            db_url=cls.db_url,
            regen=True  # Force regeneration for first test to ensure fresh schemas
        )
        
        # Apply decorator to test models
        cls.TestUser = cls.schema_decorator(TestUser)
        cls.TestProduct = cls.schema_decorator(TestProduct)
    
    def setUp(self):
        """Set up test environment before each test method."""
        super().setUp()
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("Skipping test: OPENAI_API_KEY environment variable must be set")
        
        # Get database connection
        self.db = common.db.get_db(self.db_url)
        
        # DON'T clear schema caches - this was causing the caching test to fail
        # Only clear them if we're testing regeneration specifically

    def tearDown(self):
        """Clean up after tests."""
        # Close all database connections
        for _, connection in common.db.DATABASES.items():
            connection.get_engine().dispose()
        common.db.DATABASES.clear()

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test environment."""
        super().tearDownClass()
        
        # Clean up test templates directory
        if hasattr(cls, 'test_templates_dir') and cls.test_templates_dir.exists():
            shutil.rmtree(cls.test_templates_dir, ignore_errors=True)

    def _extract_table_name(self, schema: str) -> str:
        """Helper method to extract table name from schema."""
        import re
        
        # Handle both "CREATE TABLE table_name" and "CREATE TABLE IF NOT EXISTS table_name"
        patterns = [
            r'CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([^\s\(]+)',
            r'CREATE\s+TABLE\s+([^\s\(]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, schema, re.IGNORECASE)
            if match:
                table_name = match.group(1).strip('`"[]')
                return table_name
        
        raise ValueError(f"Could not extract table name from schema: {schema}")

    def test_schema_generation(self):
        """Test that schema is properly generated for a model class."""
        # Debug: Check if decorator was applied
        self.assertTrue(hasattr(self.TestUser, 'get_sql_schema'), 
                      "get_sql_schema not found on TestUser")
        
        # Get schema
        schema = self.TestUser.get_sql_schema()
        print(f"\nGenerated Schema:\n{schema}\n")  # Debug output
        
        # Basic schema validation
        self.assertIsInstance(schema, str, "Schema should be a string")
        self.assertIn("CREATE TABLE", schema.upper(), "Schema should contain CREATE TABLE")
        
        # Apply schema to database
        try:
            self.db.init_schema(schema_sql=schema)
        except Exception as e:
            self.fail(f"Failed to apply schema: {str(e)}")
        
        # Verify table was created - extract actual table name from schema
        table_name = self._extract_table_name(schema)
        print(f"Expected table name: {table_name}")  # Debug output
        
        with self.db.get_engine().connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            print(f"Tables in database: {tables}")  # Debug output
            
            result = conn.execute(
                text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            )
            self.assertIsNotNone(result.fetchone(), f"Table '{table_name}' was not created")
            
            # Check table columns
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = {row[1].lower() for row in result.fetchall()}
            expected_columns = {'id', 'name', 'email', 'role', 'created_at'}
            self.assertTrue(
                expected_columns.issubset(columns),
                f"Expected columns {expected_columns} not found in {columns}"
            )
    
    def test_multiple_models(self):
        """Test that multiple decorated classes generate distinct schemas."""
        # Get schemas for both models
        user_schema = self.TestUser.get_sql_schema()
        product_schema = self.TestProduct.get_sql_schema()
        
        # Apply schemas to database
        self.db.init_schema(schema_sql=user_schema)
        self.db.init_schema(schema_sql=product_schema)
        
        # Extract table names from schemas
        user_table_name = self._extract_table_name(user_schema)
        product_table_name = self._extract_table_name(product_schema)
        
        print(f"User table name: {user_table_name}")
        print(f"Product table name: {product_table_name}")
        
        # Verify both tables were created
        with self.db.get_engine().connect() as conn:
            # Check user table
            result = conn.execute(
                text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{user_table_name}'")
            )
            self.assertIsNotNone(result.fetchone(), f"Table '{user_table_name}' was not created")
            
            # Check product table
            result = conn.execute(
                text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{product_table_name}'")
            )
            self.assertIsNotNone(result.fetchone(), f"Table '{product_table_name}' was not created")
    
    def test_schema_caching(self):
        """Test that schema is cached after first generation."""
        # Temporarily disable regeneration for this test
        original_regen = self.schema_decorator.regen
        self.schema_decorator.regen = False
        
        try:
            # First access - should generate and cache (or use existing cache)
            schema1 = self.TestUser.get_sql_schema()
            print(f"\nSchema 1:\n{schema1}\n")  # Debug output
            
            # Verify caching
            self.assertTrue(hasattr(self.TestUser, '__sql_schema__'), 
                          "__sql_schema__ attribute not set on model")
            
            # Second access - should use cached version
            schema2 = self.TestUser.get_sql_schema()
            print(f"\nSchema 2:\n{schema2}\n")  # Debug output
            
            self.assertEqual(schema1, schema2, 
                             "Cached schema should match generated schema")
        finally:
            # Restore original regeneration setting
            self.schema_decorator.regen = original_regen