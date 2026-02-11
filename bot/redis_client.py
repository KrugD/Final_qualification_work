"""Redis client wrapper for task queue, status tracking, and user history."""

import json
import os
import uuid
import time
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis

from utils.config import BotConfig


class RedisClient:
    """Async Redis client for bot operations."""
    
    QUEUE_KEY = "queue:tasks"
    PROCESSING_KEY = "queue:processing"
    TASK_PREFIX = "task:"
    USER_HISTORY_PREFIX = "user:history:"
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        self.redis = aioredis.from_url(
            BotConfig.REDIS_URL,
            decode_responses=True,
        )
        await self.redis.ping()
        print(f"Connected to Redis at {BotConfig.REDIS_URL}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    # ---- Task Queue ----
    
    async def add_task(self, user_id: int, chat_id: int, file_path: str,
                       original_filename: str, file_size: int = 0,
                       progress_message_id: int = 0) -> str:
        """Add a new task to the processing queue.
        
        Returns:
            str: task_id
        """
        task_id = str(uuid.uuid4())[:8]
        
        task_data = {
            "task_id": task_id,
            "user_id": str(user_id),
            "chat_id": str(chat_id),
            "file_path": file_path,
            "original_filename": original_filename,
            "file_size": str(file_size),
            "progress_message_id": str(progress_message_id),
            "status": "queued",
            "stage": "",
            "progress": "0",
            "created_at": datetime.now().isoformat(),
            "started_at": "",
            "completed_at": "",
            "error": "",
        }
        
        # Store task data
        await self.redis.hset(f"{self.TASK_PREFIX}{task_id}", mapping=task_data)
        # Set TTL of 24 hours
        await self.redis.expire(f"{self.TASK_PREFIX}{task_id}", 86400)
        
        # Add to queue
        await self.redis.rpush(self.QUEUE_KEY, task_id)
        
        return task_id
    
    async def get_next_task(self) -> Optional[dict]:
        """Get the next task from the queue (non-blocking).
        
        Returns:
            dict with task data, or None if queue is empty
        """
        task_id = await self.redis.lpop(self.QUEUE_KEY)
        if not task_id:
            return None
        
        task_data = await self.redis.hgetall(f"{self.TASK_PREFIX}{task_id}")
        if not task_data:
            return None
        
        # Mark as processing
        await self.redis.set(self.PROCESSING_KEY, task_id)
        await self.redis.hset(f"{self.TASK_PREFIX}{task_id}", mapping={
            "status": "processing",
            "started_at": datetime.now().isoformat(),
        })
        
        return task_data
    
    async def get_task(self, task_id: str) -> Optional[dict]:
        """Get task data by ID."""
        return await self.redis.hgetall(f"{self.TASK_PREFIX}{task_id}")
    
    async def update_task_progress(self, task_id: str, stage: str, progress: int):
        """Update task processing stage and progress."""
        await self.redis.hset(f"{self.TASK_PREFIX}{task_id}", mapping={
            "stage": stage,
            "progress": str(progress),
        })
    
    async def complete_task(self, task_id: str, success: bool = True, error: str = ""):
        """Mark task as completed."""
        status = "completed" if success else "failed"
        await self.redis.hset(f"{self.TASK_PREFIX}{task_id}", mapping={
            "status": status,
            "progress": "100" if success else "0",
            "completed_at": datetime.now().isoformat(),
            "error": error,
        })
        await self.redis.delete(self.PROCESSING_KEY)
    
    async def get_queue_position(self, task_id: str) -> int:
        """Get task position in queue (1-based). Returns 0 if processing or not in queue."""
        queue = await self.redis.lrange(self.QUEUE_KEY, 0, -1)
        try:
            return queue.index(task_id) + 1
        except ValueError:
            return 0
    
    async def get_queue_length(self) -> int:
        """Get total number of tasks in queue."""
        return await self.redis.llen(self.QUEUE_KEY)
    
    async def is_processing(self) -> bool:
        """Check if a task is currently being processed."""
        return await self.redis.exists(self.PROCESSING_KEY) > 0
    
    async def get_processing_task_id(self) -> Optional[str]:
        """Get ID of currently processing task."""
        return await self.redis.get(self.PROCESSING_KEY)
    
    async def remove_task_from_queue(self, task_id: str) -> bool:
        """Remove a task from the queue (cancel). Returns True if removed."""
        removed = await self.redis.lrem(self.QUEUE_KEY, 1, task_id)
        if removed:
            await self.redis.hset(f"{self.TASK_PREFIX}{task_id}", mapping={
                "status": "cancelled",
                "completed_at": datetime.now().isoformat(),
            })
        return removed > 0
    
    # ---- User History ----
    
    async def add_to_history(self, user_id: int, filename: str,
                              duration_min: float, num_speakers: int,
                              status: str = "completed"):
        """Add a processing record to user's history."""
        record = {
            "filename": filename,
            "date": datetime.now().isoformat(),
            "duration_min": round(duration_min, 1),
            "num_speakers": num_speakers,
            "status": status,
        }
        
        key = f"{self.USER_HISTORY_PREFIX}{user_id}"
        await self.redis.lpush(key, json.dumps(record, ensure_ascii=False))
        # Keep only last N records
        await self.redis.ltrim(key, 0, BotConfig.MAX_HISTORY_PER_USER - 1)
        # Set TTL of 30 days
        await self.redis.expire(key, 30 * 86400)
    
    async def get_history(self, user_id: int, limit: int = 10) -> list[dict]:
        """Get user's processing history."""
        key = f"{self.USER_HISTORY_PREFIX}{user_id}"
        records = await self.redis.lrange(key, 0, limit - 1)
        return [json.loads(r) for r in records]
    
    async def clear_history(self, user_id: int):
        """Clear user's processing history."""
        key = f"{self.USER_HISTORY_PREFIX}{user_id}"
        await self.redis.delete(key)
    
    # ---- User Task Management ----
    
    async def cancel_user_tasks(self, user_id: int) -> bool:
        """Cancel all queued and processing tasks for a user. Returns True if any were cancelled."""
        cancelled = False
        
        # 1. Cancel tasks in queue
        queue = await self.redis.lrange(self.QUEUE_KEY, 0, -1)
        for task_id in queue:
            task_data = await self.redis.hgetall(f"{self.TASK_PREFIX}{task_id}")
            if task_data and task_data.get("user_id") == str(user_id):
                removed = await self.redis.lrem(self.QUEUE_KEY, 1, task_id)
                if removed:
                    await self.redis.hset(f"{self.TASK_PREFIX}{task_id}", mapping={
                        "status": "cancelled",
                        "completed_at": datetime.now().isoformat(),
                    })
                    file_path = task_data.get("file_path", "")
                    if file_path and os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                    cancelled = True
        
        # 2. Cancel currently processing task if it belongs to user
        processing_task_id = await self.redis.get(self.PROCESSING_KEY)
        if processing_task_id:
            task_data = await self.redis.hgetall(f"{self.TASK_PREFIX}{processing_task_id}")
            if task_data and task_data.get("user_id") == str(user_id):
                await self.redis.hset(f"{self.TASK_PREFIX}{processing_task_id}", mapping={
                    "status": "cancelling",
                })
                cancelled = True
        
        return cancelled
    
    async def is_task_cancelled(self, task_id: str) -> bool:
        """Check if a task has been marked for cancellation."""
        task_data = await self.redis.hgetall(f"{self.TASK_PREFIX}{task_id}")
        return task_data.get("status") == "cancelling"


# Singleton instance
redis_client = RedisClient()
