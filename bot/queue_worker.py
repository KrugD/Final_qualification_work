"""Queue worker: picks tasks from Redis, runs pipeline, sends results back."""

import asyncio
import os
import tempfile
import traceback
from pathlib import Path

from aiogram import Bot
from aiogram.types import FSInputFile, BufferedInputFile
from pydub import AudioSegment

from bot.redis_client import redis_client
from bot.progress import ProgressNotifier
from bot.pdf_generator import generate_protocol_pdf
from utils.config import BotConfig


class QueueWorker:
    """Background worker that processes audio tasks from Redis queue."""
    
    def __init__(self, bot: Bot, preloaded_models: dict = None):
        self.bot = bot
        self.preloaded_models = preloaded_models or {}
        self._running = False
    
    async def start(self):
        """Start the worker loop."""
        self._running = True
        print("Queue worker started, waiting for tasks...")
        
        while self._running:
            try:
                task = await redis_client.get_next_task()
                
                if task is None:
                    # No tasks in queue, wait and check again
                    await asyncio.sleep(2)
                    continue
                
                await self._process_task(task)
                
            except Exception as e:
                print(f"Worker error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the worker loop."""
        self._running = False
        print("Queue worker stopped.")
    
    async def _process_task(self, task: dict):
        """Process a single task from the queue."""
        task_id = task["task_id"]
        chat_id = int(task["chat_id"])
        user_id = int(task["user_id"])
        file_path = task["file_path"]
        original_filename = task["original_filename"]
        
        print(f"Processing task {task_id}: {original_filename} for user {user_id}")
        
        # Create progress notifier
        notifier = ProgressNotifier(self.bot, chat_id)
        await notifier.send_initial(queue_position=0)
        await notifier.update_processing_started()
        
        wav_path = None
        
        try:
            # --- Convert to WAV if needed ---
            await notifier.update_stage("converting")
            wav_path = await self._ensure_wav(file_path)
            
            # --- Run pipeline in-memory ---
            from pipeline.pipeline import run_pipeline_in_memory
            
            # Create a sync callback bridge
            loop = asyncio.get_event_loop()
            
            def progress_callback(stage: str, percent: int):
                """Sync callback called from pipeline, bridges to async notifier."""
                asyncio.run_coroutine_threadsafe(
                    self._update_progress(task_id, stage, percent, notifier),
                    loop
                )
            
            # Run heavy pipeline in a thread
            result = await asyncio.to_thread(
                run_pipeline_in_memory,
                wav_path,
                force_clustering=False,
                progress_callback=progress_callback,
                preloaded_models=self.preloaded_models,
            )
            
            if not result["success"]:
                await notifier.update_stage("error")
                await redis_client.complete_task(task_id, success=False, error="Pipeline failed")
                await redis_client.add_to_history(
                    user_id, original_filename, 0, 0, status="failed"
                )
                return
            
            # --- Generate PDF ---
            await notifier.update_stage("pdf")
            
            pdf_buffer = generate_protocol_pdf(
                correction_df=result["correction_df"],
                diarization_df=result["diarization_df"],
                asr_df=result["asr_df"],
                audio_duration_min=result["audio_duration_min"],
                num_speakers=result["num_speakers"],
                original_filename=original_filename,
            )
            
            # --- Send results ---
            await notifier.update_stage("done")
            
            # Send PDF
            pdf_filename = f"protocol_{Path(original_filename).stem}.pdf"
            await self.bot.send_document(
                chat_id=chat_id,
                document=BufferedInputFile(
                    file=pdf_buffer.read(),
                    filename=pdf_filename,
                ),
                caption=(
                    f"📄 <b>Протокол встречи</b>\n\n"
                    f"Файл: {original_filename}\n"
                    f"Длительность: {result['audio_duration_min']:.1f} мин\n"
                    f"Спикеров: {result['num_speakers']}"
                ),
                parse_mode="HTML",
            )
            
            # Send clustering PNG if available
            if result.get("clustering_png"):
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=BufferedInputFile(
                        file=result["clustering_png"].read(),
                        filename="speaker_clusters.png",
                    ),
                    caption="📊 Визуализация кластеров спикеров",
                )
            
            # --- Save to history ---
            await redis_client.complete_task(task_id, success=True)
            await redis_client.add_to_history(
                user_id,
                original_filename,
                result["audio_duration_min"],
                result["num_speakers"],
                status="completed",
            )
            
            print(f"Task {task_id} completed successfully")
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            traceback.print_exc()
            
            await notifier.update_stage("error")
            await redis_client.complete_task(task_id, success=False, error=str(e))
            await redis_client.add_to_history(
                user_id, original_filename, 0, 0, status="failed"
            )
            
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "❌ <b>Ошибка при обработке</b>\n\n"
                        f"Файл: {original_filename}\n"
                        f"Ошибка: {str(e)[:200]}\n\n"
                        "Попробуйте отправить файл ещё раз."
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
        
        finally:
            # Cleanup temporary files
            self._cleanup(file_path, wav_path)
    
    async def _update_progress(self, task_id: str, stage: str, percent: int,
                                notifier: ProgressNotifier):
        """Update progress in Redis and Telegram message."""
        await redis_client.update_task_progress(task_id, stage, percent)
        await notifier.update_stage(stage, percent)
    
    async def _ensure_wav(self, file_path: str) -> str:
        """Convert audio file to WAV format if needed. Returns path to WAV file."""
        ext = Path(file_path).suffix.lower()
        
        if ext == ".wav":
            return file_path
        
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        
        print(f"Converting {ext} to WAV...")
        
        # Run conversion in thread (pydub is synchronous)
        def convert():
            audio = AudioSegment.from_file(file_path)
            audio.export(wav_path, format="wav")
            return wav_path
        
        return await asyncio.to_thread(convert)
    
    def _cleanup(self, *paths):
        """Remove temporary files."""
        for path in paths:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up: {path}")
                except OSError:
                    pass
