"""Progress notifier for Telegram bot -- updates messages with visual progress bar."""

import asyncio
import time
from typing import Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest


# Pipeline stages with display names and progress percentages
STAGES = {
    "queued":        {"label": "В очереди",              "icon": "🕐", "percent": 0},
    "downloading":   {"label": "Загрузка файла",         "icon": "📥", "percent": 5},
    "converting":    {"label": "Конвертация аудио",      "icon": "🔄", "percent": 8},
    "diarization":   {"label": "Диаризация спикеров",    "icon": "🎙", "percent": 15},
    "asr":           {"label": "Распознавание речи",     "icon": "📝", "percent": 35},
    "summarization": {"label": "Суммаризация текста",    "icon": "📋", "percent": 60},
    "correction":    {"label": "Коррекция текста",       "icon": "✏️", "percent": 80},
    "clustering":    {"label": "Кластеризация спикеров", "icon": "📊", "percent": 90},
    "pdf":           {"label": "Генерация PDF",          "icon": "📄", "percent": 95},
    "done":          {"label": "Готово",                  "icon": "✅", "percent": 100},
    "error":         {"label": "Ошибка",                 "icon": "❌", "percent": 0},
}

STAGE_ORDER = [
    "downloading", "converting", "diarization", "asr",
    "summarization", "correction", "clustering", "pdf", "done"
]


def _build_progress_bar(percent: int, width: int = 16) -> str:
    """Build a Unicode progress bar string."""
    filled = int(width * percent / 100)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {percent}%"


def _build_stage_list(current_stage: str) -> str:
    """Build a checklist of stages showing completed, current, and pending."""
    lines = []
    current_found = False
    
    for stage_key in STAGE_ORDER:
        info = STAGES[stage_key]
        
        if stage_key == current_stage:
            current_found = True
            lines.append(f"🔄 {info['label']}...")
        elif not current_found:
            # Already completed
            lines.append(f"✅ {info['label']}")
        else:
            # Not yet reached
            lines.append(f"⬜ {info['label']}")
    
    return "\n".join(lines)


class ProgressNotifier:
    """Manages a progress message in Telegram chat, updating it as pipeline progresses."""
    
    def __init__(self, bot: Bot, chat_id: int, message_id: Optional[int] = None):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id = message_id
        self._last_text = ""
        self._last_update_time = 0
        self._min_update_interval = 1.5  # seconds between edits to avoid rate limits
    
    async def send_initial(self, queue_position: int = 0, queue_total: int = 0) -> int:
        """Send the initial progress message.
        
        Returns:
            int: message_id of the sent message
        """
        if queue_position > 0:
            text = (
                "⏳ <b>Ваш запрос принят</b>\n\n"
                f"📍 Позиция в очереди: <b>{queue_position}</b> из {queue_total}\n"
                f"⏱ Примерное время ожидания: ~{queue_position * 5} мин\n\n"
                "Вы получите уведомление, когда обработка начнётся."
            )
        else:
            text = (
                "⏳ <b>Обработка аудио...</b>\n\n"
                f"{_build_progress_bar(5)}\n\n"
                f"{_build_stage_list('downloading')}"
            )
        
        msg = await self.bot.send_message(
            self.chat_id, text, parse_mode="HTML"
        )
        self.message_id = msg.message_id
        self._last_text = text
        return msg.message_id
    
    async def update_queue_position(self, position: int, total: int):
        """Update the queue position display."""
        text = (
            "⏳ <b>Ваш запрос в очереди</b>\n\n"
            f"📍 Позиция: <b>{position}</b> из {total}\n"
            f"⏱ Примерное время ожидания: ~{position * 5} мин\n\n"
            "Вы получите уведомление, когда обработка начнётся."
        )
        await self._edit_message(text)
    
    async def update_stage(self, stage: str, percent: Optional[int] = None):
        """Update the progress message to show current stage.
        
        Args:
            stage: Stage key from STAGES dict
            percent: Override percentage (if None, uses stage default)
        """
        if stage not in STAGES:
            return
        
        if percent is None:
            percent = STAGES[stage]["percent"]
        
        if stage == "done":
            text = (
                "✅ <b>Обработка завершена!</b>\n\n"
                f"{_build_progress_bar(100)}\n\n"
                f"{_build_stage_list('done')}\n\n"
                "📄 Отправляю результаты..."
            )
        elif stage == "error":
            text = (
                "❌ <b>Ошибка при обработке</b>\n\n"
                "Произошла ошибка во время обработки вашего аудио.\n"
                "Попробуйте отправить файл ещё раз или обратитесь к администратору."
            )
        else:
            text = (
                "⏳ <b>Обработка аудио...</b>\n\n"
                f"{_build_progress_bar(percent)}\n\n"
                f"{_build_stage_list(stage)}"
            )
        
        await self._edit_message(text)
    
    async def update_processing_started(self):
        """Notify that processing has started (was in queue before)."""
        text = (
            "🚀 <b>Обработка началась!</b>\n\n"
            f"{_build_progress_bar(5)}\n\n"
            f"{_build_stage_list('downloading')}"
        )
        await self._edit_message(text)
    
    async def _edit_message(self, text: str):
        """Edit the progress message, handling rate limits and deduplication."""
        if not self.message_id:
            return
        
        # Don't edit if text hasn't changed
        if text == self._last_text:
            return
        
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_update_time
        if elapsed < self._min_update_interval:
            await asyncio.sleep(self._min_update_interval - elapsed)
        
        try:
            await self.bot.edit_message_text(
                text=text,
                chat_id=self.chat_id,
                message_id=self.message_id,
                parse_mode="HTML",
            )
            self._last_text = text
            self._last_update_time = time.time()
        except TelegramBadRequest as e:
            if "message is not modified" not in str(e):
                print(f"Error editing progress message: {e}")
        except Exception as e:
            print(f"Error editing progress message: {e}")
