"""Telegram bot entry point: initializes bot, connects to Redis, starts worker."""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

from utils.config import BotConfig, ModelConfig
from bot.handlers import router
from bot.redis_client import redis_client
from bot.queue_worker import QueueWorker


def _validate_config():
    """Validate required configuration before starting."""
    if not BotConfig.TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN is not set in .env")
        sys.exit(1)
    
    if BotConfig.USE_LOCAL_API:
        if not BotConfig.TELEGRAM_API_ID or not BotConfig.TELEGRAM_API_HASH:
            print(
                "WARNING: TELEGRAM_API_ID / TELEGRAM_API_HASH not set.\n"
                "Local API Server requires these. Set USE_LOCAL_API=false to use cloud API.\n"
                "Falling back to standard Telegram API (20 MB file limit)."
            )
            BotConfig.USE_LOCAL_API = False


def _preload_models() -> dict:
    """Pre-load all ML models at startup to avoid loading them on each request.
    
    Returns:
        dict: Pre-loaded models for the pipeline
    """
    print("=" * 60)
    print("PRE-LOADING ML MODELS")
    print("=" * 60)
    
    models = {}
    
    try:
        print("Loading diarization model...")
        from utils.models import load_diarization_model
        models['diarization'] = load_diarization_model()
        print("  ✓ Diarization model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load diarization model: {e}")
    
    try:
        print("Loading ASR model (Whisper)...")
        from utils.models import load_asr_model
        models['asr'] = load_asr_model()
        print("  ✓ ASR model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load ASR model: {e}")
    
    try:
        print("Loading summarization model (FRED-T5)...")
        from utils.models import load_summarization_model
        model, tokenizer = load_summarization_model()
        models['summarization'] = (model, tokenizer)
        print("  ✓ Summarization model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load summarization model: {e}")
    
    try:
        print("Loading correction model (sage-m2m100)...")
        from utils.models import load_correction_model
        model, tokenizer = load_correction_model()
        models['correction'] = (model, tokenizer)
        print("  ✓ Correction model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load correction model: {e}")
    
    print("=" * 60)
    print(f"Models loaded: {len(models)}/4")
    print("=" * 60)
    
    return models


async def main():
    """Main async entry point."""
    _validate_config()
    
    # --- Pre-load models ---
    print("\nPre-loading models (this may take a few minutes)...")
    preloaded_models = _preload_models()
    
    # --- Initialize Bot ---
    if BotConfig.USE_LOCAL_API:
        print(f"\nUsing Telegram Bot API Local Server: {BotConfig.LOCAL_API_BASE_URL}")
        local_server = TelegramAPIServer.from_base(
            BotConfig.LOCAL_API_BASE_URL, is_local=True
        )
        session = AiohttpSession(api=local_server)
        bot = Bot(token=BotConfig.TELEGRAM_BOT_TOKEN, session=session)
    else:
        print("\nUsing standard Telegram Bot API (cloud)")
        bot = Bot(token=BotConfig.TELEGRAM_BOT_TOKEN)
    
    # --- Connect to Redis ---
    print("Connecting to Redis...")
    try:
        await redis_client.connect()
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        print("Make sure Redis is running (docker-compose up -d redis)")
        sys.exit(1)
    
    # --- Setup Dispatcher ---
    dp = Dispatcher()
    dp.include_router(router)
    
    # --- Start Queue Worker ---
    worker = QueueWorker(bot, preloaded_models=preloaded_models)
    worker_task = asyncio.create_task(worker.start())
    
    # --- Start Polling ---
    print("\n" + "=" * 60)
    print("BOT IS RUNNING")
    print("=" * 60)
    print("Send /start to the bot in Telegram to begin.\n")
    
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        print("\nShutting down...")
        await worker.stop()
        worker_task.cancel()
        await redis_client.close()
        await bot.session.close()
        print("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
