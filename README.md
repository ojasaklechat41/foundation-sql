# Foundation SQL (Experimental)

Foundation SQL is a Python-based assistant for generating, caching, and running SQL queries from typed Python functions. It now unifies query and schema generation behind a single decorator: `SQLQueryDecorator`.

## Installation

You can install the package directly from Github. We will publish it to PyPi once we move it to beta.

```bash
pip install git+ssh://git@github.com/think41/foundation-sql.git#egg=foundation_sql
```

## Decorator Overview

`SQLQueryDecorator` decorates a Python function (optionally typed with Pydantic models) and handles:

- SQL template generation via an OpenAI-compatible API
- Persistent template caching under `cache_dir`
- Optional automatic schema generation from Pydantic models and idempotent schema application
- Error-aware retries that repair the SQL using previous errors and the prior template

### Parameters

- `db_url: str` Database URL (e.g., `sqlite:///:memory:` or file DSN). Required.
- `schema: Optional[str]` Explicit DDL to initialize the DB.
- `schema_path: Optional[str]` Path to DDL file (alternative to `schema`).
- `auto_schema: bool` Enable automatic schema generation when no explicit `schema` is provided. Default `False`.
- `schema_models: Optional[list]` Pydantic models to derive schema from (optional if inferrable from function signature).
- `schema_validate: bool` Apply generated schema to DB. Idempotent. Default `True`.
- `schema_regen: Optional[bool]` Force schema regeneration; defaults to `regen` if not set.
- `schema_cache_namespace: Optional[str]` Namespace for schema cache key reuse.
- `nested_strategy: str` How to map nested models/enums, default `"tables"`.
- `table_name: Optional[str]` Override base table name for a single root model.
- `cache_dir: str` Directory for generated SQL templates. Default `"__sql__"`.
- `name: Optional[str]` Override template filename; default is `module.function.sql`.
- `regen: Optional[bool]` Force SQL template regeneration.
- `repair: Optional[int]` Number of error-repair retries on execution failure. Default `0`.
- `system_prompt: Optional[str]` Override system prompt text.
- `system_prompt_path: Optional[str]` Load system prompt from file.
- `base_url: Optional[str]`, `api_key: Optional[str]`, `model: Optional[str]` OpenAI-compatible LLM config.

Environment variables commonly used: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `DATABASE_URL`.

## Usage (Explicit Schema)

```python
from foundation_sql.query import SQLQueryDecorator

query = SQLQueryDecorator(
    # Required for explicit schema usage
    schema="""<CREATE TABLE ...>""",   # DDL string(s) to initialize DB
    db_url="sqlite:///:memory:",        # or file/path connection string

    # LLM backend (OpenAI-compatible)
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),

    # Common options
    cache_dir="__sql__",               # where SQL templates are cached
    name=None,                           # defaults to module.function.sql
    regen=False,                         # force regenerate SQL template
    repair=1,                            # retries with error feedback (0 = off)
)
```

Once defined, it can be used as a decorator e.g.

```
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
```

The parameter types can be Pydantic models (recommended). The function docstring guides the LLM. On first run, the SQL template is generated and stored under `cache_dir` using a namespaced filename: `module.function.sql`. Subsequent runs reuse the cached SQL unless `regen=True`.

Below is a minimal test-style example using an explicit schema:

```python
import os
from typing import List
import unittest
from foundation_sql import db, query
from pydantic import BaseModel


DB_URL = os.environ.get("DATABASE_URL", "sqlite:///:memory:")

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

query = query.SQLQueryDecorator(
    schema=TABLES_SCHEMA,
    db_url=DB_URL,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL"),
    repair=1,
)

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

class TestQuery(unittest.TestCase):

    db_url = DB_URL
    schema_sql = TABLES_SCHEMA
    schema_path = None

    def setUp(self):
        """Create a fresh database connection for each test."""
        # Re-initialize the schema for each test to ensure clean state.
        # init_schema is idempotent and ignores "already exists" errors.
        if (self.schema_sql or self.schema_path) and self.db_url:
            db.get_db(self.db_url).init_schema(schema_sql=self.schema_sql, schema_path=self.schema_path)
        else:
            raise ValueError("At least one of schema_sql, schema_path must be provided along with db_url")
        
    def test_users(self):
        users = get_users()
        self.assertEqual(len(users), 0)
        
        user = User(id="xxx", name="John Doe", email="john@example.com", role="user")
        create_user(user=user)
        
        users = get_users()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0], user)

    def tearDown(self):
        """Close the database connection after each test."""
        for _, connection in db.DATABASES.items():
            connection.get_engine().dispose()
        
        db.DATABASES.clear()
```

Running these tests will generate namespaced SQL files, e.g.:

```sql
#__sql__/tests.test_simple_query.create_user.sql
-- def create_user(user: tests.test_simple_query.User) -> int
-- Creates a new user.
-- Expects user.name, user.email and user.role to be defined
INSERT INTO `users` (
    `id`, 
    `name`, 
    `email`, 
    `role`
)
VALUES (
    {{user.id|default(None)}},
    {{user.name}},
    {{user.email}},
    {{user.role}}
);
```

```sql
#__sql__/tests.test_simple_query.get_users.sql

-- def get_users() -> List[tests.test_simple_query.User]
-- Gets all users.
SELECT 
    `id` as `id`,
    `name` as `name`,
    `email` as `email`,
    `role` as `role`
FROM 
    `users`
```

## Auto Schema Generation (No Explicit DDL)

`SQLQueryDecorator` can automatically generate a database schema from your Pydantic models and apply it, removing the need for a separate schema decorator.

Key options:
- `auto_schema=True` enable automatic schema generation when no explicit `schema` is provided.
- `schema_models=[MyModel, ...]` explicitly pass models used for schema derivation (optional if inferrable from function parameters).
- `nested_strategy="tables"` controls how nested models/enums are mapped. Default is separate tables for nested models.
- `table_name="users"` override default table name when a single root model is used.
- `schema_cache_namespace="myapp"` namespace for the schema cache key across functions/tests.
- `schema_regen=False/True` control whether to regenerate the schema prompt even if cached.
- `schema_validate=True` apply generated schema to the database (idempotent on SQLite).

Example:

```python
from pydantic import BaseModel
from foundation_sql.query import SQLQueryDecorator

class User(BaseModel):
    id: str
    name: str
    email: str
    role: str

query = SQLQueryDecorator(
    db_url="sqlite:///:memory:",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL"),
    auto_schema=True,
    schema_models=[User],             # helps inference
    schema_validate=True,             # apply schema once; safe if already applied
    schema_cache_namespace="users",  # share schema across functions
)

@query
def get_users() -> list[User]:
    """Gets all users."""
    pass
```

Notes:
- If tables already exist (created elsewhere), you can keep `schema_validate=True` — schema application is idempotent and ignores "already exists" errors — or set `schema_validate=False` to skip any schema touch.
- Functions without Pydantic models can still work: either provide `schema_models`, an explicit `schema`, or ensure the docstring clearly references existing tables/columns.

## Common Use Cases

### 1) Create and list users (single model parameter)
```python
from typing import Optional, List
from pydantic import BaseModel
from foundation_sql.query import SQLQueryDecorator
import os

class User(BaseModel):
    id: str
    name: str
    email: str
    created_at: Optional[str] = None

user_query = SQLQueryDecorator(
    db_url="sqlite:///./app.db",
    auto_schema=True,
    schema_models=[User],
    schema_cache_namespace="users",
    table_name="users",
    schema_validate=True,
    regen=False,
    repair=1,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)

@user_query
def create_user(user: User) -> User:
    """Create a new user"""
    ...

@user_query
def get_users() -> List[User]:
    """List users"""
    ...

# Calls
u = create_user(user={"id":"1","name":"A","email":"a@x.com"})
lst = get_users()
```

### 2) Get user by email (read-only)
```python
from typing import Optional

user_query_read = SQLQueryDecorator(
    db_url="sqlite:///./app.db",
    auto_schema=True,
    schema_models=[User],
    schema_cache_namespace="users",
    table_name="users",
    schema_validate=False,   # never touch schema during reads
    regen=False,
    repair=1,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)

@user_query_read
def get_user_by_email(email: str) -> Optional[User]:
    """Return one user by email or None"""
    ...

u = get_user_by_email(email="a@x.com")
```

### 3) Flat-arguments alternative
```python
@user_query
def create_user_flat(id: str, name: str, email: str) -> User:
    """Create a new user"""
    ...

create_user_flat(id="1", name="A", email="a@x.com")
```

Tip: Keep `schema_models=[User]` to strengthen auto-schema even when using flat args.

### 4) Use existing tables without schema changes
```python
existing_table_query = SQLQueryDecorator(
    db_url="sqlite:///./app.db",
    auto_schema=True,
    schema_models=[User],
    schema_cache_namespace="users",
    table_name="users",
    schema_validate=False,  # skip applying schema; use existing DB
)
```

### 5) Functions without models
```python
misc_query = SQLQueryDecorator(
    db_url="sqlite:///./app.db",
    auto_schema=False,              # don’t auto-generate schema
    schema_validate=False,
)

@misc_query
def delete_old_users(days: int) -> int:
    """Delete from users where created_at < now()-:days"""
    ...
```

## Caching, Regeneration, and Repair

- **Template filenames** are namespaced: `module.function.sql` under `cache_dir` (default `__sql__/`).
- **Regeneration**: set `regen=True` to overwrite an existing cached SQL template for a function.
- **Error repair loop**: set `repair=N` to allow N retries. On failure, the previous template and error message are fed back into the prompt to fix the query.

## Environment Variables

Required for LLM-backed generation:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

Database:
- `DATABASE_URL` (or pass `db_url` explicitly to the decorator)

Templates (optional):
- `FOUNDATION_SQL_TEMPLATE_DIR` to override where prompt templates (`*.j2`) are loaded from.

## Migration Notes

- The legacy `SQLTableSchemaDecorator` has been removed. Use `SQLQueryDecorator` with `auto_schema=True` and related options instead.
- Schema application in `foundation_sql/db.py` is idempotent, preventing failures on repeated runs.

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env_template` to `.env` and configure your environment variables:
   ```bash
   cp .env_template .env
   ```

- Run tests: `python -m unittest discover -q`

## Project Structure

- `query.py`: Main query and (optional) schema generation logic
- `db.py`: Database connection and management
- `cache.py`: Caching functionality
- `tests/`: Test suite
- `__sql__/`: Generated SQL queries
- `.env`: Environment variables
- `.env_template`: Template for environment variables
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation
