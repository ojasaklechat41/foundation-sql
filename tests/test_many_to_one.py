from typing import List, Optional
from tests import common
from pydantic import BaseModel


class Workspace(BaseModel):
    id: int
    name: str

class Task(BaseModel):
    id: int
    workspace: Workspace
    title: str
    description: Optional[str] = None

TABLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY(workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);
"""

query = common.create_query(schema=TABLES_SCHEMA)

@query
def create_workspace(name: str) -> Workspace:
    """
    Inserts a new workspace and returns the Workspace object.
    """
    pass

@query
def add_task_to_workspace(workspace: Workspace, title: str, description: Optional[str] = None) -> Task:
    """
    Inserts a new task into the workspace and returns the Task object.
    """
    pass

@query
def get_tasks_for_workspace(workspace: Workspace) -> List[Task]:
    """
    Returns all tasks for a workspace as Task objects with nested workspace.
    """
    pass

class TestWorkspaceTasks(common.DatabaseTests):
    schema_sql = TABLES_SCHEMA

    def test_workspace_tasks(self):
        # Add a workspace
        ws = create_workspace(name="Project Alpha")
        self.assertIsInstance(ws, Workspace)

        # Add tasks
        task1 = add_task_to_workspace(workspace=ws, title="Setup repo", description="Initialize git repository")
        task2 = add_task_to_workspace(workspace=ws, title="Write docs", description="Document the setup process")
        self.assertIsInstance(task1, Task)
        self.assertIsInstance(task2, Task)

        # Fetch tasks
        tasks = get_tasks_for_workspace(workspace=ws)
        self.assertEqual(len(tasks), 2)
        titles = {t.title for t in tasks}
        self.assertSetEqual(titles, {"Setup repo", "Write docs"})
        for t in tasks:
            self.assertEqual(t.workspace.id, ws.id)
            self.assertEqual(t.workspace.name, ws.name)
