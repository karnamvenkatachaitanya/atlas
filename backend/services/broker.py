import asyncio
import json
import logging
import os
from typing import Callable, Dict, List, Any, Optional

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)

class EventBroker:
    """
    An asynchronous Publish-Subscribe Message Broker.
    Uses Redis if available (for enterprise-grade messaging),
    otherwise falls back to asyncio in-memory queues (for local hackathon dev).
    """
    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._history: Dict[str, List[Any]] = {}
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._redis_task: Optional[asyncio.Task] = None
        self.use_redis = HAS_REDIS and os.environ.get("ATLAS_USE_REDIS", "1") == "1"

    async def connect(self):
        if self.use_redis:
            try:
                self._redis = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
                await self._redis.ping()
                self._pubsub = self._redis.pubsub()
                self._redis_task = asyncio.create_task(self._redis_listener())
                logger.info("Connected to Redis message broker.")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory broker.")
                self.use_redis = False

    async def _redis_listener(self):
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    topic = message["channel"].decode("utf-8")
                    data = json.loads(message["data"].decode("utf-8"))
                    
                    if topic not in self._history:
                        self._history[topic] = []
                    self._history[topic].append(data)
                    
                    if topic in self._subscribers:
                        for queue in self._subscribers[topic]:
                            await queue.put(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis listener error: {e}")

    async def publish(self, topic: str, message: Any):
        """Publish a message to a topic."""
        if self.use_redis and self._redis:
            await self._redis.publish(topic, json.dumps(message))
        else:
            # Fallback to in-memory
            if topic not in self._history:
                self._history[topic] = []
            self._history[topic].append(message)
            
            if topic in self._subscribers:
                for queue in self._subscribers[topic]:
                    await queue.put(message)

    async def subscribe(self, topic: str) -> asyncio.Queue:
        """Subscribe to a topic and return a queue to listen to."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
            if self.use_redis and self._pubsub:
                await self._pubsub.subscribe(topic)
                
        queue = asyncio.Queue()
        self._subscribers[topic].append(queue)
        return queue

    async def unsubscribe(self, topic: str, queue: asyncio.Queue):
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            if queue in self._subscribers[topic]:
                self._subscribers[topic].remove(queue)
            if not self._subscribers[topic] and self.use_redis and self._pubsub:
                await self._pubsub.unsubscribe(topic)

    async def close(self):
        if self._redis_task:
            self._redis_task.cancel()
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.aclose()

# Global singleton broker for the application
broker = EventBroker()
