"""Telegram bot handlers: commands, audio messages, callbacks."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

from aiogram import Router, Bot, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery

from bot.keyboards import (
    get_start_keyboard,
    get_help_keyboard,
    get_cancel_keyboard,
    get_history_keyboard,
)
from bot.progress import ProgressNotifier
from bot.redis_client import redis_client
from utils.config import BotConfig

router = Router()

# Temporary files directory
TEMP_DIR = os.path.join(tempfile.gettempdir(), "summarization_bot")
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================
# /start command
# ============================================================

WELCOME_TEXT = (
    "🎙 <b>Бот для суммаризации переговоров</b>\n"
    "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Отправьте мне аудиофайл или голосовое сообщение, "
    "и я подготовлю для вас <b>протокол встречи в формате PDF</b>.\n\n"
    "📌 <b>Что я умею:</b>\n"
    "├ 🎙 Разделение речи по спикерам\n"
    "├ 📝 Распознавание речи (Whisper)\n"
    "├ 📋 Суммаризация ключевых тезисов\n"
    "├ ✏️ Коррекция текста\n"
    "├ 📄 Генерация PDF-протокола\n"
    "└ 📊 Визуализация кластеров спикеров\n\n"
    "📎 <b>Поддерживаемые форматы:</b>\n"
    "WAV, MP3, OGG, FLAC, M4A, OPUS, AAC, WMA\n\n"
    "📏 <b>Макс. размер файла:</b> до 2 ГБ\n\n"
    "⬇️ <i>Просто отправьте аудиофайл в этот чат!</i>"
)


@router.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(
        WELCOME_TEXT,
        parse_mode="HTML",
        reply_markup=get_start_keyboard(),
    )


# ============================================================
# /history command
# ============================================================

@router.message(Command("history"))
async def cmd_history(message: Message):
    """Handle /history command."""
    await _show_history(message.chat.id, message.from_user.id, message)


async def _show_history(chat_id: int, user_id: int, message_or_callback):
    """Show user's processing history."""
    history = await redis_client.get_history(user_id, limit=10)
    
    if not history:
        text = (
            "📋 <b>История обработок</b>\n\n"
            "У вас пока нет обработанных файлов.\n"
            "Отправьте аудиофайл, чтобы начать!"
        )
    else:
        lines = ["📋 <b>История обработок</b>\n"]
        
        for i, record in enumerate(history, 1):
            date_str = ""
            try:
                dt = datetime.fromisoformat(record["date"])
                date_str = dt.strftime("%d.%m.%Y, %H:%M")
            except (KeyError, ValueError):
                date_str = "—"
            
            status_icon = "✅" if record.get("status") == "completed" else "❌"
            duration = record.get("duration_min", 0)
            speakers = record.get("num_speakers", "?")
            filename = record.get("filename", "—")
            
            lines.append(
                f"{i}. {status_icon} <b>{filename}</b>\n"
                f"   📅 {date_str}\n"
                f"   ⏱ {duration} мин  |  🎙 {speakers} спикеров"
            )
        
        text = "\n\n".join(lines)
    
    if isinstance(message_or_callback, Message):
        await message_or_callback.answer(text, parse_mode="HTML", reply_markup=get_history_keyboard())
    elif isinstance(message_or_callback, CallbackQuery):
        await message_or_callback.message.edit_text(text, parse_mode="HTML", reply_markup=get_history_keyboard())


# ============================================================
# /help command
# ============================================================

HELP_TEXT = (
    "📖 <b>Инструкция</b>\n"
    "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "<b>Как использовать:</b>\n\n"
    "1️⃣ Отправьте аудиофайл (документом или голосовым)\n"
    "2️⃣ Дождитесь обработки — прогресс отображается в чате\n"
    "3️⃣ Получите PDF-протокол с суммаризацией\n\n"
    "<b>Поддерживаемые форматы:</b>\n"
    "WAV, MP3, OGG, FLAC, M4A, OPUS, AAC, WMA\n\n"
    "<b>Команды:</b>\n"
    "/start — Главное меню\n"
    "/history — История обработок\n"
    "/help — Эта инструкция\n\n"
    "<b>Советы:</b>\n"
    "• Для лучшего качества используйте WAV или FLAC\n"
    "• Чем чище запись — тем точнее распознавание\n"
    "• Для записей > 50 мин автоматически строится\n"
    "  визуализация кластеров спикеров (PNG)"
)


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    await message.answer(HELP_TEXT, parse_mode="HTML", reply_markup=get_help_keyboard())


# ============================================================
# Callback handlers (inline buttons)
# ============================================================

@router.callback_query(F.data == "help")
async def cb_help(callback: CallbackQuery):
    """Handle 'help' button."""
    await callback.message.edit_text(HELP_TEXT, parse_mode="HTML", reply_markup=get_help_keyboard())
    await callback.answer()


@router.callback_query(F.data == "about")
async def cb_about(callback: CallbackQuery):
    """Handle 'about' button."""
    text = (
        "ℹ️ <b>О боте</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Бот для автоматической суммаризации аудиозаписей\n"
        "переговоров и встреч.\n\n"
        "<b>Технологии:</b>\n"
        "├ 🎙 pyannote (диаризация спикеров)\n"
        "├ 📝 OpenAI Whisper (распознавание речи)\n"
        "├ 📋 FRED-T5 (суммаризация на русском)\n"
        "├ ✏️ sage-m2m100 (коррекция текста)\n"
        "└ 📊 sklearn (кластеризация)\n\n"
        "Все модели работают локально.\n"
        "Ваши данные не отправляются на внешние серверы."
    )
    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=get_help_keyboard())
    await callback.answer()


@router.callback_query(F.data == "history")
async def cb_history(callback: CallbackQuery):
    """Handle 'history' button."""
    await _show_history(callback.message.chat.id, callback.from_user.id, callback)
    await callback.answer()


@router.callback_query(F.data == "back_to_start")
async def cb_back_to_start(callback: CallbackQuery):
    """Handle 'back' button - return to start screen."""
    await callback.message.edit_text(
        WELCOME_TEXT, parse_mode="HTML", reply_markup=get_start_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data.startswith("cancel:"))
async def cb_cancel_task(callback: CallbackQuery):
    """Handle task cancellation."""
    task_id = callback.data.split(":", 1)[1]
    
    removed = await redis_client.remove_task_from_queue(task_id)
    
    if removed:
        await callback.message.edit_text(
            "🚫 <b>Обработка отменена</b>\n\n"
            "Ваш запрос удалён из очереди.",
            parse_mode="HTML",
        )
        await callback.answer("Отменено")
    else:
        await callback.answer("Задача уже обрабатывается, отмена невозможна", show_alert=True)


# ============================================================
# Audio file handlers
# ============================================================

@router.message(F.audio)
async def handle_audio(message: Message, bot: Bot):
    """Handle audio file messages (sent as audio)."""
    await _process_audio_message(message, bot, message.audio.file_id,
                                  message.audio.file_name or "audio.mp3",
                                  message.audio.file_size or 0)


@router.message(F.voice)
async def handle_voice(message: Message, bot: Bot):
    """Handle voice messages."""
    await _process_audio_message(message, bot, message.voice.file_id,
                                  "voice_message.ogg",
                                  message.voice.file_size or 0)


@router.message(F.document)
async def handle_document(message: Message, bot: Bot):
    """Handle document messages (for WAV/FLAC files sent as documents)."""
    doc = message.document
    filename = doc.file_name or "document"
    ext = Path(filename).suffix.lower()
    
    if ext not in BotConfig.SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(BotConfig.SUPPORTED_AUDIO_FORMATS))
        await message.reply(
            f"⚠️ Неподдерживаемый формат: <b>{ext}</b>\n\n"
            f"Поддерживаемые форматы: {supported}",
            parse_mode="HTML",
        )
        return
    
    await _process_audio_message(message, bot, doc.file_id, filename, doc.file_size or 0)


@router.message(F.video_note)
async def handle_video_note(message: Message, bot: Bot):
    """Handle video notes (круглые видеосообщения) - extract audio."""
    await _process_audio_message(message, bot, message.video_note.file_id,
                                  "video_note.mp4",
                                  message.video_note.file_size or 0)


async def _process_audio_message(message: Message, bot: Bot,
                                   file_id: str, filename: str, file_size: int):
    """Common handler for all audio input types.
    
    Downloads the file, validates it, adds to queue, and notifies user.
    """
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    # Check file size
    max_size_bytes = BotConfig.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        await message.reply(
            f"⚠️ Файл слишком большой: {file_size / 1024 / 1024:.0f} МБ\n"
            f"Максимальный размер: {BotConfig.MAX_FILE_SIZE_MB} МБ",
            parse_mode="HTML",
        )
        return
    
    # Download file
    status_msg = await message.reply("📥 Загрузка файла...")
    
    try:
        file = await bot.get_file(file_id)
        
        # Ensure unique filename
        safe_filename = Path(filename).stem[:50] + Path(filename).suffix
        local_path = os.path.join(TEMP_DIR, f"{user_id}_{message.message_id}_{safe_filename}")
        
        await bot.download_file(file.file_path, local_path)
        
        print(f"Downloaded {filename} ({file_size} bytes) -> {local_path}")
        
    except Exception as e:
        await status_msg.edit_text(
            f"❌ Ошибка при загрузке файла: {str(e)[:200]}",
        )
        return
    
    # Add task to Redis queue
    task_id = await redis_client.add_task(
        user_id=user_id,
        chat_id=chat_id,
        file_path=local_path,
        original_filename=filename,
        file_size=file_size,
    )
    
    # Check queue position
    queue_length = await redis_client.get_queue_length()
    is_processing = await redis_client.is_processing()
    
    position = queue_length  # Our task is the last one
    total_in_queue = queue_length + (1 if is_processing else 0)
    
    # Delete the "downloading" status message
    try:
        await status_msg.delete()
    except Exception:
        pass
    
    # Send progress message
    notifier = ProgressNotifier(bot, chat_id)
    
    if position > 1 or (position == 1 and is_processing):
        # There are tasks ahead of us
        msg_id = await notifier.send_initial(
            queue_position=position,
            queue_total=total_in_queue,
        )
    else:
        # We're next (or only task)
        msg_id = await notifier.send_initial(queue_position=0)
    
    print(f"Task {task_id} added to queue. Position: {position}, Queue length: {queue_length}")


# ============================================================
# Fallback handler for unsupported messages
# ============================================================

@router.message()
async def handle_unknown(message: Message):
    """Handle any other messages."""
    await message.reply(
        "🤔 Я принимаю только аудиофайлы и голосовые сообщения.\n\n"
        "Отправьте аудио или используйте /help для справки.",
        parse_mode="HTML",
    )
