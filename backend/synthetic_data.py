import asyncio
import json
import logging
import random
import os
from typing import Optional

from backend.services.broker import broker

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_MODEL_URL = os.environ.get(
    "ATLAS_HF_API_URL", 
    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
)

class SyntheticDataStreamer:
    """
    Simulates external industry conditions continuously.
    Connects to Hugging Face API to generate realistic market news and events,
    which are broadcasted to the system via the event broker.
    """
    def __init__(self, interval_seconds: float = 5.0):
        self.interval_seconds = interval_seconds
        self.running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self.running: return
        self.running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Synthetic Data Streamer started (interval={self.interval_seconds}s)")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Synthetic Data Streamer stopped.")

    async def _run_loop(self):
        try:
            while self.running:
                await asyncio.sleep(self.interval_seconds)
                
                # Randomly decide to emit an event
                if random.random() < 0.4:
                    event = await self._generate_market_event()
                    await broker.publish("market_events", event)
                    logger.info(f"Broadcasted Market Event: {event['title']}")
        except asyncio.CancelledError:
            pass

    async def _generate_market_event(self) -> dict:
        """Generates an event via HF API or uses fallback."""
        event_types = ["COMPETITOR_ANNOUNCEMENT", "MACRO_ECONOMICS", "VIRAL_TWEET", "SECURITY_VULNERABILITY"]
        chosen_type = random.choice(event_types)
        
        prompt = (
            f"You are a financial news API.\n"
            f"Generate a realistic short headline and impact score (-10 to +10) for a startup simulation.\n"
            f"Category: {chosen_type}\n"
            "Respond ONLY with a JSON dictionary in this format:\n"
            '{"title": "The headline...", "impact": -5}'
        )

        if HAS_AIOHTTP and HF_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 100, "temperature": 0.8, "return_full_text": False}
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(HF_MODEL_URL, headers=headers, json=payload, timeout=5.0) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            gen_text = data[0]["generated_text"]
                            start = gen_text.find("{")
                            end = gen_text.rfind("}")
                            if start != -1 and end != -1:
                                result = json.loads(gen_text[start:end+1])
                                result["type"] = chosen_type
                                return result
            except Exception as e:
                logger.warning(f"Synthetic Streamer API call failed: {e}")

        # Local fallback data
        fallbacks = {
            "COMPETITOR_ANNOUNCEMENT": {"title": "Rival startup launches identical feature.", "impact": -3},
            "MACRO_ECONOMICS": {"title": "Interest rates hike slightly, cooling VC market.", "impact": -5},
            "VIRAL_TWEET": {"title": "Tech influencer praises your product UI!", "impact": 6},
            "SECURITY_VULNERABILITY": {"title": "Zero-day exploit found in popular open source library.", "impact": -8}
        }
        
        event = fallbacks[chosen_type]
        event["type"] = chosen_type
        # Add random noise to impact
        event["impact"] += random.randint(-2, 2)
        return event
