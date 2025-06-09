"""
Tests for the database query parser functions.
"""

import unittest
from datetime import datetime
from enum import Enum
from typing import Optional

from foundation_sql import db
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List
from tests import common

# --- Test SQL Schema ---
TEST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    context_window INTEGER,
    max_tokens INTEGER,
    created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agent (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    instructions TEXT,
    type VARCHAR(50),
    model_id VARCHAR(36),
    created_at TIMESTAMP,
    FOREIGN KEY(model_id) REFERENCES model(id)
);
CREATE TABLE IF NOT EXISTS task (
    id VARCHAR(36) PRIMARY KEY,
    task_no INTEGER,
    title VARCHAR(255),
    description TEXT,
    status VARCHAR(50),
    agent_id VARCHAR(36),
    parent_task_id VARCHAR(36),
    created_at TIMESTAMP,
    FOREIGN KEY(agent_id) REFERENCES agent(id),
    FOREIGN KEY(parent_task_id) REFERENCES task(id)
);
"""

# --- Pydantic Models and Enums (previously from schema) ---
class Model(BaseModel):
    id: str
    name: str
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    created_at: Optional[datetime] = None

class AgentType(str, Enum):
    GENERALIST = "generalist"
    SPECIALIST = "specialist"

class Agent(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    type: AgentType
    model: Optional[Model] = None
    created_at: Optional[datetime] = None

class TaskStatus(str, Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    ASSIGNED_FOR_WORK = "assigned_for_work"

class Task(BaseModel):
    id: str
    task_no: Optional[int] = None
    title: str
    description: Optional[str] = None
    status: TaskStatus
    agent: Optional[Agent] = None
    parent_task: Optional['Task'] = None
    created_at: Optional[datetime] = None

Task.model_rebuild()


class TestDbParser(common.DatabaseTests):
    """Tests for the db.parse_query_to_pydantic function."""

    db_url = "sqlite:///:memory:"
    schema_sql = TEST_SCHEMA_SQL

    def test_parse_basic_model(self):
        """Test parsing a basic model without nested fields or enums."""
        # Create test data that would come from a query
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Test Model",
            "context_window": 4096,
            "max_tokens": 1024,
            "created_at": "2025-01-01 12:00:00"
        }
        
        # Parse the data into a Model
        result = db.parse_query_to_pydantic(data, Model)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(result.name, "Test Model")
        self.assertEqual(result.context_window, 4096)
        self.assertEqual(result.max_tokens, 1024)
        self.assertEqual(result.created_at, datetime.strptime("2025-01-01 12:00:00", "%Y-%m-%d %H:%M:%S"))

    def test_parse_with_enum(self):
        """Test parsing a model with enum fields."""
        # Create test data with enum values as it would come from a database query
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Test Agent",
            "description": "A test agent",
            "instructions": "Test instructions",
            "type": "specialist",  # This is an enum value
            "model.id": "00000000-0000-0000-0000-000000000002",
            "model.name": "Test Model",
            "model.context_window": 4096,
            "model.max_tokens": 1024,
            "created_at": "2025-01-01 12:00:00"
        }
        
        # Parse the data into an Agent
        result = db.parse_query_to_pydantic(data, Agent)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(result.name, "Test Agent")
        self.assertEqual(result.type, AgentType.SPECIALIST)
        self.assertIsInstance(result.type, AgentType)
        self.assertIsNotNone(result.model)
        self.assertEqual(result.model.name, "Test Model")

    def test_parse_with_optional_enum(self):
        """Test parsing a model with optional enum fields."""
        # Create a simple test class with optional enum
        class TestStatus(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"
            
        class TestModel(BaseModel):
            name: str
            status: Optional[TestStatus] = None
        
        # Test with enum value present
        data1 = {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Test Item",
            "status": "active",
            "created_at": "2025-01-01 12:00:00"
        }
        
        result1 = db.parse_query_to_pydantic(data1, TestModel)
        self.assertEqual(result1.status, TestStatus.ACTIVE)
        self.assertIsInstance(result1.status, TestStatus)
        
        # Test with enum value absent
        data2 = {
            "id": "00000000-0000-0000-0000-000000000002",
            "name": "Test Item 2",
            "created_at": "2025-01-01 12:00:00"
        }

        result2 = db.parse_query_to_pydantic(data2, TestModel)
        self.assertIsNone(result2.status)

    def test_parse_with_nested_model(self):
        """Test parsing a model with a nested model."""
        # Create test data with a nested model (Agent with Model)
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Test Agent",
            "description": "A test agent",
            "instructions": "Test instructions",
            "type": "specialist",
            "model.id": "00000000-0000-0000-0000-000000000002",
            "model.name": "Test Model",
            "model.context_window": 4096,
            "model.max_tokens": 1024,
            "created_at": "2025-01-01 12:00:00"
        }
        
        # Parse the data into an Agent
        result = db.parse_query_to_pydantic(data, Agent)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(result.name, "Test Agent")
        
        # Check nested model
        self.assertIsNotNone(result.model)
        self.assertEqual(result.model.id, "00000000-0000-0000-0000-000000000002")
        self.assertEqual(result.model.name, "Test Model")
        self.assertEqual(result.model.context_window, 4096)
        self.assertEqual(result.model.max_tokens, 1024)

    def test_parse_with_optional_nested_model(self):
        """Test parsing a model with an optional nested model that is present."""
        # Create a simple test that doesn't involve complex nested models
        # First, let's create a simpler test class with an optional nested model
        class SimpleModel(BaseModel):
            name: str
            value: int
            
        class ContainerModel(BaseModel):
            id: str
            title: str
            nested: Optional[SimpleModel] = None
        
        # Create test data for a container with a nested model
        data = {
            "id": "test-id-123",
            "title": "Test Container",
            "nested.name": "Test Nested",
            "nested.value": 42
        }
        
        # Parse the data into a ContainerModel
        result = db.parse_query_to_pydantic(data, ContainerModel)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "test-id-123")
        self.assertEqual(result.title, "Test Container")
        self.assertIsNotNone(result.nested)
        self.assertEqual(result.nested.name, "Test Nested")
        self.assertEqual(result.nested.value, 42)

    def test_parse_without_optional_nested_model(self):
        """Test parsing a model with an optional nested model that is not present."""
        # Create test data for a Task without an Agent
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "task_no": 1,
            "title": "Test Task",
            "description": "A test task",
            "status": "new",
            "created_at": "2025-01-01 12:00:00"
        }
        
        # Parse the data into a Task
        result = db.parse_query_to_pydantic(data, Task)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(result.title, "Test Task")
        self.assertEqual(result.status, TaskStatus.NEW)
        
        # Check that agent is None
        self.assertIsNone(result.agent)

    def test_parse_empty_data(self):
        """Test parsing with empty data returns None."""
        # Test with None
        result1 = db.parse_query_to_pydantic(None, Model)
        self.assertIsNone(result1)
        
        # Test with empty dict
        result2 = db.parse_query_to_pydantic({}, Model)
        self.assertIsNone(result2)

    def test_parse_complex_nested_structure(self):
        """Test parsing a complex nested structure with multiple levels."""
        # Create test data for a Task with a parent Task and an Agent
        # This simulates the flat dictionary that would come from a database query
        # with joined tables and aliased columns using double-underscore notation
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "task_no": 2,
            "title": "Subtask",
            "description": "A subtask",
            "status": "in_progress",
            "created_at": "2025-01-01 12:00:00",
            
            # Agent fields with double-underscore notation
            "agent.id": "00000000-0000-0000-0000-000000000002",
            "agent.name": "Test Agent",
            "agent.description": "A test agent",
            "agent.instructions": "Test instructions",
            "agent.type": "specialist",
            "agent.created_at": "2025-01-01 11:00:00",
            
            # Agent's model fields with double-underscore notation
            "agent.model.id": "00000000-0000-0000-0000-000000000004",
            "agent.model.name": "Agent Model",
            "agent.model.context_window": 4096,
            "agent.model.max_tokens": 1024,
            "agent.model.created_at": "2025-01-01 10:00:00",
            
            # Parent task fields with double-underscore notation
            "parent_task.id": "00000000-0000-0000-0000-000000000003",
            "parent_task.task_no": 1,
            "parent_task.title": "Parent Task",
            "parent_task.description": "A parent task",
            "parent_task.status": "assigned_for_work",
            "parent_task.created_at": "2025-01-01 09:00:00"
        }
        
        # Parse the data into a Task - this should work by properly handling the nested fields
        # with double-underscore notation
        result = db.parse_query_to_pydantic(data, Task)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(result.title, "Subtask")
        self.assertEqual(result.status, TaskStatus.IN_PROGRESS)
        
        # Check nested agent
        self.assertIsNotNone(result.agent)
        self.assertEqual(result.agent.id, "00000000-0000-0000-0000-000000000002")
        self.assertEqual(result.agent.name, "Test Agent")
        self.assertIsNotNone(result.agent.model)
        self.assertEqual(result.agent.model.name, "Agent Model")
        
        # Check parent task
        self.assertIsNotNone(result.parent_task)
        self.assertEqual(result.parent_task.id, "00000000-0000-0000-0000-000000000003")
        self.assertEqual(result.parent_task.title, "Parent Task")
        self.assertEqual(result.parent_task.status, TaskStatus.ASSIGNED_FOR_WORK)


if __name__ == '__main__':
    unittest.main()
