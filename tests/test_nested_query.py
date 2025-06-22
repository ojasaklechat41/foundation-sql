from typing import List, Optional
from tests import common
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Profile(BaseModel):
    id: Optional[int] = None
    bio: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    address: Optional[str] = None  # Store as text for simplicity

class UserWithProfile(BaseModel):
    id: str
    name: str
    email: str
    role: str
    profile: Optional[Profile] = None

# FIXED SCHEMA - separate tables that match the generated SQL
TABLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'user', 'guest')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(36) NOT NULL,
    bio TEXT,
    age INTEGER,
    gender VARCHAR(50),
    address TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
"""

query = common.create_query(schema=TABLES_SCHEMA)

@query
def get_users_with_profile() -> List[UserWithProfile]:
    """
    Gets all users with their profiles.
    """
    pass

@query
def create_user_with_profile(user: UserWithProfile) -> int:
    """
    Creates a new user with a profile.
    """
    pass

class TestNestedQuery(common.DatabaseTests):

    schema_sql = TABLES_SCHEMA
        
    def test_nested_object_creation_and_retrieval(self):
        # Initially, no users
        users = get_users_with_profile()
        self.assertEqual(len(users), 0)
        
        # Create a user with a simplified profile (no nested Address for now)
        profile = Profile(bio="Software Engineer", age=30, gender="female")
        user = UserWithProfile(
            id="nested_user_1", 
            name="Jane Doe", 
            email="jane@example.com", 
            role="user", 
            profile=profile
        )
        
        # Store the user
        create_user_with_profile(user=user)
        
        # Retrieve and verify
        retrieved_users = get_users_with_profile()
        self.assertEqual(len(retrieved_users), 1)
        
        retrieved_user = retrieved_users[0]
        self.assertEqual(retrieved_user.id, "nested_user_1")
        self.assertEqual(retrieved_user.name, "Jane Doe")
        self.assertEqual(retrieved_user.email, "jane@example.com")
        self.assertEqual(retrieved_user.role, "user")
        
        # Check nested profile
        self.assertIsNotNone(retrieved_user.profile)
        self.assertEqual(retrieved_user.profile.bio, "Software Engineer")
    
    def test_nested_object_with_partial_data(self):
        # Create a user with a partial profile
        user = UserWithProfile(
            id="nested_user_2", 
            name="John Smith", 
            email="john@example.com", 
            role="guest", 
            profile=Profile(bio="Data Scientist")
        )
        
        # Store the user
        create_user_with_profile(user=user)
        
        # Retrieve and verify
        retrieved_users = get_users_with_profile()
        self.assertEqual(len(retrieved_users), 1)
        
        # Find the newly added user
        retrieved_user = next(u for u in retrieved_users if u.id == "nested_user_2")
        
        self.assertEqual(retrieved_user.id, "nested_user_2")
        self.assertEqual(retrieved_user.name, "John Smith")
        self.assertEqual(retrieved_user.email, "john@example.com")
        self.assertEqual(retrieved_user.role, "guest")
        
        # Check partial profile
        self.assertIsNotNone(retrieved_user.profile)
        self.assertEqual(retrieved_user.profile.bio, "Data Scientist")