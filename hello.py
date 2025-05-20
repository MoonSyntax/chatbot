# task_manager.py

from typing import List, Dict, Optional
import uuid
import datetime

class Task:
    def __init__(self, title: str, description: str, due_date: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.created_at = datetime.datetime.now()
        self.due_date = due_date
        self.completed = False

    def mark_complete(self):
        self.completed = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date,
            "completed": self.completed
        }

class TaskManager:
    def __init__(self):
        self.tasks: List[Task] = []

    def add_task(self, title: str, description: str, due_date: Optional[str] = None):
        task = Task(title, description, due_date)
        self.tasks.append(task)
        return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def complete_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if task:
            task.mark_complete()
            return True
        return False

    def list_tasks(self, include_completed: bool = True) -> List[Dict]:
        result = []
        for task in self.tasks:
            if include_completed or not task.completed:
                result.append(task.to_dict())
        return result

    def delete_task(self, task_id: str) -> bool:
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                del self.tasks[i]
                return True
        return False

def create_sample_tasks(manager: TaskManager):
    for i in range(20):
        manager.add_task(
            title=f"Task {i+1}",
            description="This is a sample task for testing the task manager.",
            due_date="2025-12-31"
        )

def main():
    manager = TaskManager()
    create_sample_tasks(manager)
    print("All Tasks (Initial):")
    for task in manager.list_tasks():
        print(task)

    first_task_id = manager.tasks[0].id
    print("\nCompleting first task...")
    manager.complete_task(first_task_id)

    print("\nIncomplete Tasks:")
    for task in manager.list_tasks(include_completed=False):
        print(task)

if __name__ == "__main__":
    main()
