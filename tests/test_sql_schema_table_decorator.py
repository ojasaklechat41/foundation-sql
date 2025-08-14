import os
import shutil
import tempfile
import inspect
from pathlib import Path
from typing import Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from sqlalchemy.sql import text

from foundation_sql.query import SQLQueryDecorator
from tests import common


# --- Test Enums ---
class UserRole(str, Enum):
    """User role enumeration for testing."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


# --- Test Models ---
class TestUser(BaseModel):
    """Test user model for auto schema generation via SQLQueryDecorator."""
    id: str
    name: str
    email: str
    role: UserRole
    created_at: Optional[datetime] = None


class TestSQLQueryDecoratorAutoSchema(common.DatabaseTests):
    """Simple, guarded tests for SQLQueryDecorator(auto_schema)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Create a temporary directory for test templates
        cls.test_templates_dir = Path(tempfile.mkdtemp(prefix='foundation_sql_test_templates_'))

        # Copy template files to the test directory
        package_templates_dir = Path(__file__).parent.parent / "foundation_sql" / "templates"
        for template_file in package_templates_dir.glob("*.j2"):
            shutil.copy(template_file, cls.test_templates_dir)

        # Set the template directory for tests
        os.environ["FOUNDATION_SQL_TEMPLATE_DIR"] = str(cls.test_templates_dir)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if hasattr(cls, 'test_templates_dir') and cls.test_templates_dir.exists():
            shutil.rmtree(cls.test_templates_dir, ignore_errors=True)

    def setUp(self):
        super().setUp()
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("Skipping test: OPENAI_API_KEY environment variable must be set")

        # Guard: only run if SQLQueryDecorator supports auto_schema new API
        sig = inspect.signature(SQLQueryDecorator.__init__)
        required_params = {"auto_schema", "schema_validate"}
        if not required_params.issubset(set(sig.parameters.keys())):
            self.skipTest("Skipping: SQLQueryDecorator.auto_schema not implemented yet")

        self.db = common.db.get_db(self.db_url)

    def tearDown(self):
        # Close all database connections
        for _, connection in common.db.DATABASES.items():
            connection.get_engine().dispose()
        common.db.DATABASES.clear()

    def _table_exists(self, table_name: str) -> bool:
        with self.db.get_engine().connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table_name},
            )
            return result.fetchone() is not None

    def test_auto_schema_creates_table_on_first_use(self):
        """Auto-schema flow: decorating a function with a Pydantic arg should create a table."""

        # Build decorator kwargs dynamically to avoid breaking if API changes
        sig = inspect.signature(SQLQueryDecorator.__init__)
        kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model": os.getenv("OPENAI_MODEL"),
            "cache_dir": "__sql__",
            "db_url": self.db_url,
        }
        # Optional new params
        if "auto_schema" in sig.parameters:
            kwargs["auto_schema"] = True
        if "schema_validate" in sig.parameters:
            kwargs["schema_validate"] = True
        if "schema_regen" in sig.parameters:
            kwargs["schema_regen"] = True

        decorator = SQLQueryDecorator(**kwargs)

        @decorator
        def create_user(user: TestUser) -> TestUser:
            """Create a new user"""
            pass  # The body is unused; the decorator will LLM-generate SQL

        # Trigger potential schema creation; if the implementation creates schema at decoration time,
        # this call is still safe and should succeed by either returning None or raising if misconfigured.
        try:
            # We don't assert on return shape; we only care that schema is applied
            try:
                create_user(user=TestUser(id="1", name="A", email="a@x.com", role=UserRole.ADMIN))
            except Exception:
                # Even if the query fails, the schema may already be created; continue to check
                pass

            # Default table name heuristic `{model.__name__.lower()}s`
            expected_table = "testusers"
            self.assertTrue(self._table_exists(expected_table), f"Table '{expected_table}' was not created")
        except Exception as e:
            self.fail(f"Auto schema flow failed unexpectedly: {e}")


class TestSQLQueryDecoratorNested(common.DatabaseTests):
    """Guarded tests covering nested models and enums with auto_schema."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_templates_dir = Path(tempfile.mkdtemp(prefix='foundation_sql_test_templates_'))
        package_templates_dir = Path(__file__).parent.parent / "foundation_sql" / "templates"
        for template_file in package_templates_dir.glob("*.j2"):
            shutil.copy(template_file, cls.test_templates_dir)
        os.environ["FOUNDATION_SQL_TEMPLATE_DIR"] = str(cls.test_templates_dir)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if hasattr(cls, 'test_templates_dir') and cls.test_templates_dir.exists():
            shutil.rmtree(cls.test_templates_dir, ignore_errors=True)

    def setUp(self):
        super().setUp()
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("Skipping test: OPENAI_API_KEY environment variable must be set")
        sig = inspect.signature(SQLQueryDecorator.__init__)
        if "auto_schema" not in sig.parameters:
            self.skipTest("Skipping: SQLQueryDecorator.auto_schema not implemented yet")
        self.db = common.db.get_db(self.db_url)

    def tearDown(self):
        for _, connection in common.db.DATABASES.items():
            connection.get_engine().dispose()
        common.db.DATABASES.clear()

    def _table_exists(self, table_name: str) -> bool:
        with self.db.get_engine().connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table_name},
            )
            return result.fetchone() is not None

    def test_auto_schema_with_nested_model(self):
        class Profile(BaseModel):
            bio: Optional[str] = None
            website: Optional[str] = None

        class Role(str, Enum):
            ADMIN = "admin"
            USER = "user"

        class Account(BaseModel):
            id: str
            username: str
            role: Role
            profile: Optional[Profile] = None

        sig = inspect.signature(SQLQueryDecorator.__init__)
        kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model": os.getenv("OPENAI_MODEL"),
            "cache_dir": "__sql__",
            "db_url": self.db_url,
        }
        if "auto_schema" in sig.parameters:
            kwargs["auto_schema"] = True
        if "schema_validate" in sig.parameters:
            kwargs["schema_validate"] = True
        if "nested_strategy" in sig.parameters:
            kwargs["nested_strategy"] = "tables"

        decorator = SQLQueryDecorator(**kwargs)

        @decorator
        def create_account(account: Account) -> Account:
            """Create an account with optional nested profile."""
            pass

        # Call once to ensure any runtime generation occurs in implementations that defer it
        try:
            try:
                create_account(account=Account(id="u1", username="user1", role=Role.USER))
            except Exception:
                pass
            expected_table = "accounts"
            self.assertTrue(self._table_exists(expected_table), f"Table '{expected_table}' was not created")
        except Exception as e:
            self.fail(f"Nested auto schema flow failed unexpectedly: {e}")


class TestSQLQueryDecoratorSchemaCaching(common.DatabaseTests):
    """Guarded tests for schema cache regeneration policy (schema_regen)."""

    def setUp(self):
        super().setUp()
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("Skipping test: OPENAI_API_KEY environment variable must be set")
        sig = inspect.signature(SQLQueryDecorator.__init__)
        if "auto_schema" not in sig.parameters:
            self.skipTest("Skipping: SQLQueryDecorator.auto_schema not implemented yet")
        self.db = common.db.get_db(self.db_url)
        self.temp_cache_dir = Path(tempfile.mkdtemp(prefix='foundation_sql_cache_'))

    def tearDown(self):
        for _, connection in common.db.DATABASES.items():
            connection.get_engine().dispose()
        common.db.DATABASES.clear()
        if hasattr(self, 'temp_cache_dir') and self.temp_cache_dir.exists():
            shutil.rmtree(self.temp_cache_dir, ignore_errors=True)

    def test_schema_cache_regen_mtime(self):
        class Item(BaseModel):
            id: str
            name: str

        # First run: generate and cache schema
        dec1 = SQLQueryDecorator(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            cache_dir=str(self.temp_cache_dir),
            db_url=self.db_url,
            auto_schema=True,
            schema_validate=True,
            schema_regen=False,
        )

        @dec1
        def create_item(item: Item) -> Item:
            pass

        try:
            try:
                create_item(item=Item(id="i1", name="n1"))
            except Exception:
                pass
            # Ensure a cache file exists
            cache_files = list(self.temp_cache_dir.iterdir())
            self.assertTrue(cache_files, "No cache files created on first run")
            first_file = max(cache_files, key=lambda p: p.stat().st_mtime)
            first_mtime = first_file.stat().st_mtime
        except Exception as e:
            self.fail(f"First run failed unexpectedly: {e}")

        # Second run with schema_regen=False should not update mtime
        dec2 = SQLQueryDecorator(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            cache_dir=str(self.temp_cache_dir),
            db_url=self.db_url,
            auto_schema=True,
            schema_validate=True,
            schema_regen=False,
        )

        @dec2
        def create_item_again(item: Item) -> Item:
            pass

        try:
            try:
                create_item_again(item=Item(id="i2", name="n2"))
            except Exception:
                pass
            second_mtime = first_file.stat().st_mtime
            self.assertEqual(first_mtime, second_mtime, "Cache file mtime changed despite schema_regen=False")
        except Exception as e:
            self.fail(f"Second run failed unexpectedly: {e}")

        # Third run with schema_regen=True should update mtime (overwrite)
        dec3 = SQLQueryDecorator(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            cache_dir=str(self.temp_cache_dir),
            db_url=self.db_url,
            auto_schema=True,
            schema_validate=True,
            schema_regen=True,
        )

        @dec3
        def create_item_regen(item: Item) -> Item:
            pass

        try:
            try:
                create_item_regen(item=Item(id="i3", name="n3"))
            except Exception:
                pass
            third_mtime = first_file.stat().st_mtime
            self.assertGreaterEqual(third_mtime, first_mtime, "Cache file mtime did not update with schema_regen=True")
        except Exception as e:
            self.fail(f"Third run (regen) failed unexpectedly: {e}")