import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from tests import common
from pydantic import BaseModel, Field


class User(BaseModel):
    id: Optional[int] = Field(default=None)  # Make id optional for auto-increment
    name: str
    email: str
    role: str

TABLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'user', 'guest')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
)
"""

query = common.create_query(schema=TABLES_SCHEMA)

@query
def get_users() -> List[User]:
    """
    Gets all users.
    """
    pass

@query
def create_user(user: User) -> int:
    """
    Creates a new user and returns the user ID.
    """
    pass

class TestQuery(common.DatabaseTests):
    schema_sql = TABLES_SCHEMA
    
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
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test environment."""
        # Clean up test templates directory
        if hasattr(cls, 'test_templates_dir') and cls.test_templates_dir.exists():
            shutil.rmtree(cls.test_templates_dir)
        
    def test_users(self):
        # First, verify templates are being used
        template_dir = os.environ.get("FOUNDATION_SQL_TEMPLATE_DIR")
        if template_dir:
            self.assertTrue(os.path.exists(os.path.join(template_dir, "query_prompt.j2")),
                           "Query prompt template not found in test directory")
        
        users = get_users()
        self.assertEqual(len(users), 0)
        
        user = User(name="John Doe", email="john@example.com", role="user")
        user_id = create_user(user=user)
        
        # Verify the user was created
        self.assertIsInstance(user_id, int)
        self.assertGreater(user_id, 0)
        
        users = get_users()
        self.assertEqual(len(users), 1)
        
        # Check the retrieved user
        retrieved_user = users[0]
        self.assertIsNotNone(retrieved_user.id)
        self.assertGreater(retrieved_user.id, 0)
        self.assertEqual(retrieved_user.name, "John Doe")
        self.assertEqual(retrieved_user.email, "john@example.com")
        self.assertEqual(retrieved_user.role, "user")