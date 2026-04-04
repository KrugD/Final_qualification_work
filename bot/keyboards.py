"""Keyboards for the Telegram bot."""

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)


# ============================================================
# Reply Keyboard (persistent buttons below input field)
# ============================================================

def get_main_reply_keyboard() -> ReplyKeyboardMarkup:
    """Main persistent keyboard below the input field."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="📋 История"),
                KeyboardButton(text="❌ Отменить обработку"),
            ],
            [
                KeyboardButton(text="🗑 Очистить историю"),
                KeyboardButton(text="📖 Помощь"),
            ],
        ],
        resize_keyboard=True,
        input_field_placeholder="Отправьте аудиофайл...",
    )


# ============================================================
# Inline Keyboards (buttons attached to messages)
# ============================================================

def get_start_keyboard() -> InlineKeyboardMarkup:
    """Inline buttons for start message."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📖 Инструкция", callback_data="help"),
            InlineKeyboardButton(text="ℹ️ О боте", callback_data="about"),
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


def get_clear_history_confirm_keyboard() -> InlineKeyboardMarkup:
    """Confirmation buttons for clearing history."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да, очистить", callback_data="clear_history_confirm"),
            InlineKeyboardButton(text="◀️ Отмена", callback_data="back_to_start"),
        ],
    ])
