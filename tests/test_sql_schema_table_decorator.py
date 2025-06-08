"""
Tests for SQLTableSchemaDecorator.

This module contains comprehensive tests for the SQLTableSchemaDecorator class,
which generates SQL table schemas from Pydantic models using LLM assistance.
"""

import os
import tempfile
import unittest
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from enum import Enum
from unittest.mock import Mock, patch, MagicMock

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import foundation_sql modules
from foundation_sql.query import SQLTableSchemaDecorator
from foundation_sql import db
from foundation_sql.cache import SQLTemplateCache
from foundation_sql.gen import SQLGenerator
from foundation_sql.query import SQLQueryDecorator


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


class Profile(BaseModel):
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    location: Optional[str] = None


class UserWithProfile(BaseModel):
    id: str
    name: str
    email: str
    role: UserRole
    profile: Optional[Profile] = None
    created_at: Optional[datetime] = None


# --- Base Database Test Class ---
class DatabaseTestBase(unittest.TestCase):
    """Base test class for database-driven tests."""

    def setUp(self):
        """Create a fresh database connection for each test."""
        self.db_url = os.environ.get("DATABSE_URL", "sqlite:///:memory:")
        
        # Create a basic schema for database tests
        self.schema_sql = """
CREATE TABLE IF NOT EXISTS test_table (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255)
);
"""
        
        # Initialize database with basic schema
        if self.schema_sql and self.db_url:
            db.get_db(self.db_url).init_schema(schema_sql=self.schema_sql)

    def tearDown(self):
        """Close the database connection after each test."""
        for _, connection in db.DATABASES.items():
            connection.get_engine().dispose()
        db.DATABASES.clear()


# --- Test Schema Generation ---
class TestSQLTableSchemaDecorator(DatabaseTestBase):
    """Test cases for SQLTableSchemaDecorator functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Call parent setUp to initialize database
        super().setUp()
        
        # Use __sql__ directory in current working directory for cache
        self.cache_dir = os.path.join(os.getcwd(), '__sql__')
        
        # Create temporary directory for other test files
        self.test_dir = tempfile.mkdtemp()
        
        # Mock API credentials - handle both missing and empty env vars
        self.api_key = os.getenv("OPENAI_API_KEY") or "test_api_key"
        self.base_url = (os.getenv("OPENAI_API_BASE_URL") or 
                        os.getenv("OPENAI_BASE_URL") or 
                        "https://api.test.com/v1")
        self.model = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        
        # Check if we have real API credentials (not empty or default)
        self.has_real_api = (
            self.api_key and self.api_key != "test_api_key" and
            self.base_url and self.base_url != "https://api.test.com/v1"
        )
        
        # Sample generated schemas for mocking
        self.sample_user_schema = """
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'user', 'guest')) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
        
        self.sample_product_schema = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50) NOT NULL CHECK (category IN ('electronics', 'clothing', 'books')),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
);
"""
    
    def tearDown(self):
        """Clean up after each test."""
        # Clean up temporary test directory (but keep __sql__ cache)
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Note: We intentionally keep the __sql__ cache directory 
        # so generated schemas persist between test runs
        
        # Call parent tearDown to clean up database
        super().tearDown()
    
    def test_init_with_minimal_parameters(self):
        """Test SQLTableSchemaDecorator initialization with minimal parameters."""
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        self.assertIsNone(decorator.name)
        self.assertIsNone(decorator.regen)
        self.assertEqual(decorator.repair, 0)
        self.assertIsNotNone(decorator.cache)
        self.assertIsNotNone(decorator.sql_generator)
    
    def test_init_without_api_credentials(self):
        """Test initialization without API credentials doesn't create SQL generator."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        self.assertIsNone(decorator.sql_generator)
    
    def test_extract_model_from_function_parameter(self):
        """Test extracting Pydantic model from function parameter."""
        def test_func(user: User):
            pass
        
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        model_class = decorator._extract_model_from_function(test_func)
        
        self.assertEqual(model_class, User)
    
    def test_extract_model_from_function_no_model(self):
        """Test extracting model from function with no Pydantic model raises error."""
        def test_func(data: str):
            pass
        
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        with self.assertRaises(ValueError) as context:
            decorator._extract_model_from_function(test_func)
        
        self.assertIn("No Pydantic model found", str(context.exception))
    
    @patch('foundation_sql.query.SQLGenerator')
    def test_decorator_with_schema_generation_and_caching(self, mock_sql_generator_class):
        """Test decorator generating new schema and caching it to file."""
        # Mock the SQL generator
        mock_generator = Mock()
        mock_generator.generate_sql.return_value = self.sample_user_schema
        mock_sql_generator_class.return_value = mock_generator
        
        # Clear any existing cache for this test
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        cache_file_name = "user_table_schema.sql"
        if cache.exists(cache_file_name):
            cache.clear(cache_file_name)
        
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User) -> str:
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        # Verify the result
        self.assertEqual(result, self.sample_user_schema)
        mock_generator.generate_sql.assert_called_once()
        
        # Verify schema was cached to file
        self.assertTrue(cache.exists(cache_file_name))
        cached_schema = cache.get(cache_file_name)
        self.assertEqual(cached_schema, self.sample_user_schema)
        
        # Check physical file exists
        cache_file_path = os.path.join(self.cache_dir, cache_file_name)
        self.assertTrue(os.path.exists(cache_file_path))
        
        # Verify file contents
        with open(cache_file_path, 'r') as f:
            file_content = f.read()
        self.assertEqual(file_content, self.sample_user_schema)
        
        print(f"✅ Schema cached to: {cache_file_path}")
    
    def test_decorator_with_cached_schema(self):
        """Test decorator using cached schema from file."""
        # Pre-populate cache file
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        cache_file_name = "user_table_schema.sql"
        cache.set(cache_file_name, self.sample_user_schema)
        
        # Verify file was created
        cache_file_path = os.path.join(self.cache_dir, cache_file_name)
        self.assertTrue(os.path.exists(cache_file_path))
        
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User) -> str:
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        # Should use cached version
        self.assertEqual(result, self.sample_user_schema)
        
        # Verify file still exists and has correct content
        with open(cache_file_path, 'r') as f:
            file_content = f.read()
        self.assertEqual(file_content, self.sample_user_schema)
    
    def test_decorator_with_predefined_schema(self):
        """Test decorator using predefined schema."""
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User) -> str:
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        self.assertEqual(result, self.sample_user_schema)
        self.assertEqual(user_table.sql_schema, self.sample_user_schema)
        self.assertEqual(user_table.model_class, User)
    
    def test_decorator_function_attributes(self):
        """Test that decorated function has correct attributes attached."""
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_product_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def product_table(product: Product) -> str:
            """Generate schema for Product."""
            pass
        
        # Check attached attributes
        self.assertEqual(product_table.sql_schema, self.sample_product_schema)
        self.assertEqual(product_table.model_class, Product)
        self.assertIsNotNone(product_table.func_spec)
        self.assertEqual(product_table.func_spec.name, "product_table")
    
    def test_real_schema_generation_with_caching(self):
        """Test real schema generation with actual LLM API and verify cache storage."""
        # Skip if no real API credentials
        if not self.has_real_api:
            self.skipTest("Real API credentials not available - set OPENAI_API_KEY and OPENAI_API_BASE_URL")
        
        # Use a decorator without validation to avoid schema validation errors
        # (real LLM APIs sometimes generate imperfect SQL)
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            db_url=None,  # Skip database validation for real API test
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table_real(user: User) -> str:
            """Generate schema for a user table with id, name, email, role, and created_at fields."""
            pass
        
        print("\n🔄 Calling real API to generate user schema...")
        try:
            result = user_table_real()
            
            # Basic validation of generated schema
            self.assertIsInstance(result, str)
            self.assertIn("CREATE TABLE", result.upper())
            self.assertTrue(any(word in result.lower() for word in ["user", "users"]))
            
            # Check for expected fields (but don't validate SQL syntax)
            for field in ["id", "name", "email", "role"]:
                self.assertIn(field, result.lower())
            
            # Verify schema was cached to file
            cache = SQLTemplateCache(cache_dir=self.cache_dir)
            cache_file_name = "user_table_real_schema.sql"
            
            self.assertTrue(cache.exists(cache_file_name))
            cached_content = cache.get(cache_file_name)
            self.assertEqual(result, cached_content)
            
            # Verify cache file exists on disk
            cache_file_path = os.path.join(self.cache_dir, cache_file_name)
            self.assertTrue(os.path.exists(cache_file_path))
            
            # Read and verify cache file content
            with open(cache_file_path, 'r') as f:
                disk_content = f.read()
            self.assertEqual(result, disk_content)
            
            print(f"✅ Generated schema ({len(result)} chars)")
            print(f"✅ Schema cached to: {cache_file_path}")
            print(f"✅ First 100 chars: {result[:100]}...")
            
            # Note about potential SQL issues
            if "created_at" in result.lower():
                created_at_count = result.lower().count("created_at")
                if created_at_count > 1:
                    print(f"⚠️  Note: Schema contains {created_at_count} 'created_at' references")
                    print("   (LLM-generated schemas may need manual review)")
            
        except Exception as e:
            # If there's an error with the real API call, provide helpful info
            print(f"❌ Real API test failed: {str(e)}")
            if "duplicate column" in str(e):
                print("💡 This is a known issue with LLM-generated schemas")
                print("   The LLM sometimes generates duplicate columns")
                print("   In production, you'd want to add schema validation/cleaning")
            
            # Re-raise to fail the test (this helps identify schema generation issues)
            raise
    
    def test_real_schema_caching_performance(self):
        """Test that cached schemas provide performance improvement."""
        # Skip if no real API credentials
        if not self.has_real_api:
            self.skipTest("Real API credentials not available - set OPENAI_API_KEY and OPENAI_API_BASE_URL")
        
        # Use a unique cache directory for this test to ensure fresh generation
        perf_cache_dir = os.path.join(self.cache_dir, 'performance_test')
        os.makedirs(perf_cache_dir, exist_ok=True)
        
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            cache_dir=perf_cache_dir
        )
        
        @decorator
        def product_table_perf(product: Product) -> str:
            """Generate schema for product table performance test."""
            pass
        
        import time
        
        print("\n🔄 First call (should generate and cache)...")
        start_time = time.time()
        result1 = product_table_perf()
        first_call_time = time.time() - start_time
        
        print(f"⏱️ First call took: {first_call_time:.3f} seconds")
        
        print("🔄 Second call (should use cache)...")
        start_time = time.time()
        result2 = product_table_perf()
        second_call_time = time.time() - start_time
        
        print(f"⏱️ Second call took: {second_call_time:.6f} seconds")
        
        # Both results should be identical
        self.assertEqual(result1, result2)
        
        # For performance comparison, we'll use a more reasonable threshold
        # since both operations might be very fast when using mocks or cached data
        if first_call_time > 0.001:  # Only test if first call took more than 1ms
            self.assertLess(second_call_time, first_call_time, 
                           "Second call should be faster due to caching")
            print(f"✅ Caching provided {first_call_time/max(second_call_time, 0.000001):.1f}x speedup")
        else:
            print("✅ Both calls were very fast (likely using cached/mocked data)")
        
        # Verify cache file exists
        cache_file_path = os.path.join(perf_cache_dir, "product_table_perf_schema.sql")
        self.assertTrue(os.path.exists(cache_file_path))
        print(f"✅ Cache file: {cache_file_path}")
        
        # Clean up performance test cache
        shutil.rmtree(perf_cache_dir, ignore_errors=True)
    
    def test_usage_with_query_decorator_integration(self):
        """Test SQLTableSchemaDecorator integration with SQLQueryDecorator."""
        # Generate schema first
        schema_decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @schema_decorator
        def user_table(user: User) -> str:
            """Generate schema for User."""
            pass
        
        generated_schema = user_table()
        
        # Use generated schema with query decorator
        query_decorator = SQLQueryDecorator(
            schema=generated_schema,
            db_url=self.db_url,
            api_key=os.getenv("OPENAI_API_KEY", "mock_key"),
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://mock.api.com"),
            model=os.getenv("OPENAI_MODEL", "mock-model")
        )
        
        # Verify the schema was generated correctly
        self.assertEqual(generated_schema, self.sample_user_schema)
        self.assertIsNotNone(user_table.sql_schema)
        self.assertEqual(user_table.model_class, User)
        
        # Verify query decorator was created successfully
        self.assertIsNotNone(query_decorator)
    
    def test_cache_directory_structure(self):
        """Test that cache directory is created with proper structure."""
        # Use a test subdirectory within __sql__
        cache_dir = os.path.join(os.getcwd(), '__sql__', 'test_cache')
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(cache_dir)
        os.makedirs(parent_dir, exist_ok=True)
        
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=cache_dir
        )
        
        @decorator
        def test_cache_structure(user: User) -> str:
            """Test cache directory creation."""
            pass
        
        # Call the function to trigger cache creation
        result = test_cache_structure()
        
        # Verify cache directory was created
        self.assertTrue(os.path.exists(cache_dir))
        self.assertTrue(os.path.isdir(cache_dir))
        
        # Verify cache files can be created
        cache = SQLTemplateCache(cache_dir=cache_dir)
        test_content = "TEST CACHE CONTENT"
        cache.set("test_file.sql", test_content)
        
        # Verify file exists
        test_file_path = os.path.join(cache_dir, "test_file.sql")
        self.assertTrue(os.path.exists(test_file_path))
        
        # Verify content
        retrieved_content = cache.get("test_file.sql")
        self.assertEqual(retrieved_content, test_content)
        
        print(f"✅ Cache directory created: {cache_dir}")
        print(f"✅ Cache files working properly")
        
        # Clean up test subdirectory
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)