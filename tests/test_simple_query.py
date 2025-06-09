from typing import List
from tests import common
from pydantic import BaseModel


class User(BaseModel):
    id: str
    name: str
    email: str
    role: str

TABLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
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
    Creates a new user.
    """
    pass

class TestQuery(common.DatabaseTests):

    schema_sql = TABLES_SCHEMA
        
    def test_users(self):
        users = get_users()
        self.assertEqual(len(users), 0)
        
        user = User(id="xxx", name="John Doe", email="john@example.com", role="user")
        create_user(user=user)
        
        users = get_users()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0], user)
