"""Inline keyboards for the Telegram bot."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def get_start_keyboard() -> InlineKeyboardMarkup:
    """Main start menu keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📖 Инструкция", callback_data="help"),
            InlineKeyboardButton(text="ℹ️ О боте", callback_data="about"),
        ],
        [
            InlineKeyboardButton(text="📋 Моя история", callback_data="history"),
        ],
    ])


def get_help_keyboard() -> InlineKeyboardMarkup:
    """Back button for help screen."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_to_start")],
    ])


def get_cancel_keyboard(task_id: str) -> InlineKeyboardMarkup:
    """Cancel button shown during queue waiting."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="❌ Отменить", callback_data=f"cancel:{task_id}")],
    ])


def get_clustering_keyboard(task_id: str) -> InlineKeyboardMarkup:
    """Button to request clustering visualization after receiving PDF."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="📊 Визуализация кластеров спикеров",
            callback_data=f"clustering:{task_id}"
        )],
    ])


def get_history_keyboard() -> InlineKeyboardMarkup:
    """Back button for history screen."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_to_start")],
    ])
