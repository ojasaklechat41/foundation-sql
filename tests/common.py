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

# Import foundation_sql modules
from foundation_sql.query import SQLTableSchemaDecorator
from foundation_sql import db
from foundation_sql.cache import SQLTemplateCache
from foundation_sql.gen import SQLGenerator
from tests.common import DatabaseTests


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


class Order(BaseModel):
    id: int
    user_id: str
    product_id: int
    quantity: int
    total_amount: float
    order_date: Optional[datetime] = None


# --- Test Schema Generation ---
class TestSQLTableSchemaDecorator(DatabaseTests):
    """Test cases for SQLTableSchemaDecorator functionality."""
    
    # Provide a dummy schema to satisfy DatabaseTests requirement
    schema_sql = """
CREATE TABLE IF NOT EXISTS test_table (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255)
);
"""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Call parent setUp to initialize database
        super().setUp()
        
        # Create temporary directory for cache
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, '__test_sql__')
        
        # Mock API credentials
        self.api_key = os.getenv("OPENAI_API_KEY", "test_api_key")
        self.base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.test.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
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
        
        self.sample_profile_schema = """
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bio TEXT,
    avatar_url VARCHAR(500),
    location VARCHAR(255)
);
"""
    
    def tearDown(self):
        """Clean up after each test."""
        # Clean up test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
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
    
    def test_init_with_all_parameters(self):
        """Test SQLTableSchemaDecorator initialization with all parameters."""
        decorator = SQLTableSchemaDecorator(
            name="custom_schema",
            regen=True,
            repair=3,
            schema="CUSTOM SCHEMA",
            system_prompt="Custom prompt",
            db_url=self.db_url,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            cache_dir=self.cache_dir
        )
        
        self.assertEqual(decorator.name, "custom_schema")
        self.assertTrue(decorator.regen)
        self.assertEqual(decorator.repair, 3)
        self.assertEqual(decorator.schema, "CUSTOM SCHEMA")
        self.assertEqual(decorator.db_url, self.db_url)
    
    def test_init_without_api_credentials(self):
        """Test initialization without API credentials doesn't create SQL generator."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        self.assertIsNone(decorator.sql_generator)
    
    def test_load_file_existing(self):
        """Test loading content from an existing file."""
        test_file = os.path.join(self.test_dir, 'test_schema.sql')
        test_content = "CREATE TABLE test_table (id INT);"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        content = decorator.load_file(test_file)
        
        self.assertEqual(content, test_content)
    
    def test_load_file_nonexistent(self):
        """Test loading content from a non-existent file returns None."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        content = decorator.load_file("nonexistent_file.sql")
        
        self.assertIsNone(content)
    
    def test_extract_model_from_function_parameter(self):
        """Test extracting Pydantic model from function parameter."""
        def test_func(user: User):
            pass
        
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        model_class = decorator._extract_model_from_function(test_func)
        
        self.assertEqual(model_class, User)
    
    def test_extract_model_from_function_type_hint(self):
        """Test extracting Pydantic model from function type hints."""
        def test_func() -> User:
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
    def test_generate_schema_from_model(self, mock_sql_generator_class):
        """Test generating schema from Pydantic model."""
        # Mock the SQL generator
        mock_generator = Mock()
        mock_generator.generate_sql.return_value = self.sample_user_schema
        mock_sql_generator_class.return_value = mock_generator
        
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        result = decorator._generate_schema_from_model(
            User, 
            "test_func", 
            "Test function"
        )
        
        self.assertEqual(result, self.sample_user_schema)
        mock_generator.generate_sql.assert_called_once()
    
    def test_generate_schema_from_model_no_generator(self):
        """Test generating schema without SQL generator raises error."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        with self.assertRaises(ValueError) as context:
            decorator._generate_schema_from_model(
                User, 
                "test_func", 
                "Test function"
            )
        
        self.assertIn("No SQL generator available", str(context.exception))
    
    @patch('foundation_sql.db.get_db')
    def test_validate_schema_success(self, mock_get_db):
        """Test successful schema validation."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        
        decorator = SQLTableSchemaDecorator(
            db_url=self.db_url,
            cache_dir=self.cache_dir
        )
        
        # Should not raise any exception
        decorator._validate_schema(self.sample_user_schema)
        mock_db.init_schema.assert_called_once_with(schema_sql=self.sample_user_schema)
    
    @patch('foundation_sql.db.get_db')
    def test_validate_schema_failure(self, mock_get_db):
        """Test schema validation failure."""
        mock_db = Mock()
        mock_db.init_schema.side_effect = Exception("Schema error")
        mock_get_db.return_value = mock_db
        
        decorator = SQLTableSchemaDecorator(
            db_url=self.db_url,
            cache_dir=self.cache_dir
        )
        
        with self.assertRaises(ValueError) as context:
            decorator._validate_schema("INVALID SCHEMA")
        
        self.assertIn("Schema validation failed", str(context.exception))
    
    def test_validate_schema_no_db_url(self):
        """Test schema validation without db_url does nothing."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        # Should not raise any exception
        decorator._validate_schema(self.sample_user_schema)
    
    def test_decorator_with_predefined_schema(self):
        """Test decorator using predefined schema."""
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        self.assertEqual(result, self.sample_user_schema)
        self.assertEqual(user_table.sql_schema, self.sample_user_schema)
        self.assertEqual(user_table.model_class, User)
    
    def test_decorator_with_cached_schema(self):
        """Test decorator using cached schema."""
        # Pre-populate cache
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        cache.set("user_table_schema.sql", self.sample_user_schema)
        
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        self.assertEqual(result, self.sample_user_schema)
    
    @patch('foundation_sql.query.SQLGenerator')
    def test_decorator_with_schema_generation(self, mock_sql_generator_class):
        """Test decorator generating new schema."""
        # Mock the SQL generator
        mock_generator = Mock()
        mock_generator.generate_sql.return_value = self.sample_user_schema
        mock_sql_generator_class.return_value = mock_generator
        
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        self.assertEqual(result, self.sample_user_schema)
        mock_generator.generate_sql.assert_called_once()
        
        # Check if schema was cached
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        cached_schema = cache.get("user_table_schema.sql")
        self.assertEqual(cached_schema, self.sample_user_schema)
    
    @patch('foundation_sql.query.SQLGenerator')
    def test_decorator_with_regeneration(self, mock_sql_generator_class):
        """Test decorator with forced regeneration."""
        # Pre-populate cache with old schema
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        cache.set("user_table_schema.sql", "OLD SCHEMA")
        
        # Mock the SQL generator
        mock_generator = Mock()
        mock_generator.generate_sql.return_value = self.sample_user_schema
        mock_sql_generator_class.return_value = mock_generator
        
        decorator = SQLTableSchemaDecorator(
            regen=True,  # Force regeneration
            api_key=self.api_key,
            base_url=self.base_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        self.assertEqual(result, self.sample_user_schema)
        mock_generator.generate_sql.assert_called_once()
        
        # Check if new schema was cached
        cached_schema = cache.get("user_table_schema.sql")
        self.assertEqual(cached_schema, self.sample_user_schema)
    
    def test_decorator_with_enum_model(self):
        """Test decorator with model containing enums."""
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table_with_enum(user: User):
            """Generate schema for User with enum role."""
            pass
        
        result = user_table_with_enum()
        
        self.assertEqual(result, self.sample_user_schema)
        self.assertEqual(user_table_with_enum.model_class, User)
        
        # Verify that the model has enum fields
        self.assertTrue(hasattr(user_table_with_enum.model_class, 'role'))
        self.assertEqual(user_table_with_enum.model_class.model_fields['role'].annotation, UserRole)
    
    def test_decorator_with_nested_model(self):
        """Test decorator with model containing nested fields."""
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_with_profile_table(user: UserWithProfile):
            """Generate schema for UserWithProfile."""
            pass
        
        result = user_with_profile_table()
        
        self.assertEqual(result, self.sample_user_schema)
        self.assertEqual(user_with_profile_table.model_class, UserWithProfile)
        
        # Verify nested model structure
        self.assertTrue(hasattr(user_with_profile_table.model_class, 'profile'))
    
    def test_decorator_with_custom_name(self):
        """Test decorator with custom schema name."""
        decorator = SQLTableSchemaDecorator(
            name="custom_user_schema.sql",
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        result = user_table()
        
        self.assertEqual(result, self.sample_user_schema)
        
        # Check if schema was cached with custom name
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        self.assertTrue(cache.exists("custom_user_schema.sql"))
    
    def test_decorator_function_attributes(self):
        """Test that decorated function has correct attributes attached."""
        decorator = SQLTableSchemaDecorator(
            schema=self.sample_product_schema,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def product_table(product: Product):
            """Generate schema for Product."""
            pass
        
        # Check attached attributes
        self.assertEqual(product_table.sql_schema, self.sample_product_schema)
        self.assertEqual(product_table.model_class, Product)
        self.assertIsNotNone(product_table.func_spec)
        self.assertEqual(product_table.func_spec.name, "product_table")
    
    @unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "API credentials not available")
    def test_real_schema_generation_user(self):
        """Test real schema generation for User model with actual LLM API."""
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            db_url=self.db_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def user_table(user: User):
            """Generate schema for a user table with id, name, email, role, and created_at fields."""
            pass
        
        result = user_table()
        
        # Basic validation of generated schema
        self.assertIsInstance(result, str)
        self.assertIn("CREATE TABLE", result.upper())
        self.assertTrue(any(word in result.lower() for word in ["user", "users"]))
        
        # Check for expected fields
        for field in ["id", "name", "email", "role"]:
            self.assertIn(field, result.lower())
        
        # Check that schema was cached
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        self.assertTrue(cache.exists("user_table_schema.sql"))
    
    @unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "API credentials not available")
    def test_real_schema_generation_product(self):
        """Test real schema generation for Product model with actual LLM API."""
        decorator = SQLTableSchemaDecorator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            db_url=self.db_url,
            cache_dir=self.cache_dir
        )
        
        @decorator
        def product_table(product: Product):
            """Generate schema for a product table with id, name, price, category, description, and is_active fields."""
            pass
        
        result = product_table()
        
        # Basic validation of generated schema
        self.assertIsInstance(result, str)
        self.assertIn("CREATE TABLE", result.upper())
        self.assertTrue(any(word in result.lower() for word in ["product", "products"]))
        
        # Check for expected fields
        for field in ["id", "name", "price", "category"]:
            self.assertIn(field, result.lower())
    
    def test_multiple_model_schemas(self):
        """Test generating schemas for multiple different models."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            return self.sample_user_schema
        
        @decorator  
        def product_table(product: Product):
            """Generate schema for Product."""
            return self.sample_product_schema
        
        # Override the decorator's schema loading for testing
        user_table.sql_schema = self.sample_user_schema
        product_table.sql_schema = self.sample_product_schema
        
        user_result = user_table()
        product_result = product_table()
        
        self.assertEqual(user_result, self.sample_user_schema)
        self.assertEqual(product_result, self.sample_product_schema)
        self.assertEqual(user_table.model_class, User)
        self.assertEqual(product_table.model_class, Product)
    
    def test_schema_caching_behavior(self):
        """Test that schemas are properly cached and reused."""
        # Pre-populate cache
        cache = SQLTemplateCache(cache_dir=self.cache_dir)
        cache.set("user_table_schema.sql", self.sample_user_schema)
        
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        @decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        # Should load from cache
        result = user_table()
        
        # Verify it matches cached content
        cached_content = cache.get("user_table_schema.sql")
        self.assertEqual(result, cached_content)
        self.assertEqual(result, self.sample_user_schema)
    
    def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        decorator = SQLTableSchemaDecorator(cache_dir=self.cache_dir)
        
        # Test with function that has no Pydantic model
        def invalid_func(data: str):
            pass
        
        with self.assertRaises(ValueError):
            decorator._extract_model_from_function(invalid_func)
        
        # Test with non-existent schema file
        content = decorator.load_file("/nonexistent/path/schema.sql")
        self.assertIsNone(content)
        
        # Test schema generation without SQL generator
        with self.assertRaises(ValueError):
            decorator._generate_schema_from_model(User, "test", "test")
    
    def test_usage_with_query_decorator_integration(self):
        """Test SQLTableSchemaDecorator integration with SQLQueryDecorator."""
        from tests.common import create_query
        
        # Generate schema first
        schema_decorator = SQLTableSchemaDecorator(
            schema=self.sample_user_schema,
            cache_dir=self.cache_dir
        )
        
        @schema_decorator
        def user_table(user: User):
            """Generate schema for User."""
            pass
        
        generated_schema = user_table()
        
        # Use generated schema with query decorator
        query_decorator = create_query(generated_schema)
        
        # Verify the schema was generated correctly
        self.assertEqual(generated_schema, self.sample_user_schema)
        self.assertIsNotNone(user_table.sql_schema)
        self.assertEqual(user_table.model_class, User)


if __name__ == '__main__':
    unittest.main(verbosity=2)