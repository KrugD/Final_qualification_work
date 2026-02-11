"""PDF protocol generator for meeting summarization results."""

import io
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from fpdf import FPDF


# Path to DejaVu fonts (bundled with most Linux systems)
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
]

_FONT_BOLD_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"),
]


def _find_font(paths: list[str]) -> Optional[str]:
    """Find first existing font file from list of paths."""
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


class ProtocolPDF(FPDF):
    """Custom PDF class for meeting protocol with header and footer."""
    
    def __init__(self, title: str = "Протокол встречи"):
        super().__init__()
        self._title = title
        self._font_loaded = False
        self._setup_fonts()
    
    def _setup_fonts(self):
        """Register DejaVu fonts for Cyrillic support."""
        font_path = _find_font(_FONT_SEARCH_PATHS)
        font_bold_path = _find_font(_FONT_BOLD_SEARCH_PATHS)
        
        if font_path:
            self.add_font("DejaVu", "", font_path, uni=True)
            self._font_loaded = True
        
        if font_bold_path:
            self.add_font("DejaVu", "B", font_bold_path, uni=True)
        elif font_path:
            # Use regular as bold fallback
            self.add_font("DejaVu", "B", font_path, uni=True)
    
    def _use_font(self, style: str = "", size: int = 11):
        """Set font with fallback."""
        if self._font_loaded:
            self.set_font("DejaVu", style, size)
        else:
            self.set_font("Helvetica", style, size)
    
    def header(self):
        """Page header with title and line."""
        self._use_font("B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, self._title, ln=True, align="L")
        self.set_draw_color(52, 73, 94)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)
    
    def footer(self):
        """Page footer with page number and generation date."""
        self.set_y(-15)
        self._use_font("", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Стр. {self.page_no()}/{{nb}}", align="C")


def generate_protocol_pdf(
    correction_df: pd.DataFrame,
    diarization_df: Optional[pd.DataFrame] = None,
    asr_df: Optional[pd.DataFrame] = None,
    audio_duration_min: float = 0,
    num_speakers: int = 0,
    original_filename: str = "audio",
) -> io.BytesIO:
    """Generate a PDF meeting protocol from pipeline results.
    
    Args:
        correction_df: DataFrame with corrected summaries (speaker, corrected_summary)
        diarization_df: Optional DataFrame with diarization results
        asr_df: Optional DataFrame with ASR results (for word counts)
        audio_duration_min: Audio file duration in minutes
        num_speakers: Number of unique speakers detected
        original_filename: Name of the original audio file
        
    Returns:
        BytesIO: PDF file buffer
    """
    pdf = ProtocolPDF(title="Протокол встречи")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ---- Title ----
    pdf._use_font("B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.ln(5)
    pdf.cell(0, 14, "Протокол встречи", ln=True, align="C")
    pdf.ln(2)
    
    # ---- Decorative line ----
    pdf.set_draw_color(52, 152, 219)
    pdf.set_line_width(1)
    y = pdf.get_y()
    pdf.line(60, y, pdf.w - 60, y)
    pdf.ln(8)
    
    # ---- Metadata block ----
    pdf._use_font("", 10)
    pdf.set_text_color(100, 100, 100)
    
    now = datetime.now()
    meta_lines = [
        f"Дата обработки: {now.strftime('%d.%m.%Y, %H:%M')}",
        f"Исходный файл: {original_filename}",
        f"Длительность: {audio_duration_min:.1f} мин",
        f"Обнаружено спикеров: {num_speakers}",
    ]
    
    # Calculate total words if ASR data available
    if asr_df is not None and not asr_df.empty and "word_count" in asr_df.columns:
        total_words = int(asr_df["word_count"].sum())
        meta_lines.append(f"Всего слов распознано: {total_words}")
    
    for line in meta_lines:
        pdf.cell(0, 6, line, ln=True, align="C")
    
    pdf.ln(8)
    
    # ---- Separator ----
    pdf.set_draw_color(189, 195, 199)
    pdf.set_line_width(0.3)
    y = pdf.get_y()
    pdf.line(10, y, pdf.w - 10, y)
    pdf.ln(6)
    
    # ---- Speaker summaries ----
    if correction_df is not None and not correction_df.empty:
        for idx, row in correction_df.iterrows():
            speaker = row.get("speaker", f"Спикер {idx}")
            summary = row.get("corrected_summary", row.get("summary", ""))
            
            # Speaker header with colored background
            pdf.set_fill_color(52, 152, 219)
            pdf.set_text_color(255, 255, 255)
            pdf._use_font("B", 12)
            pdf.cell(0, 9, f"  {speaker}", ln=True, fill=True)
            pdf.ln(3)
            
            # Speaker statistics
            if diarization_df is not None and not diarization_df.empty:
                speaker_segments = diarization_df[diarization_df["speaker"] == speaker]
                if not speaker_segments.empty:
                    total_time = speaker_segments["duration"].sum()
                    num_segments = len(speaker_segments)
                    
                    pdf._use_font("", 9)
                    pdf.set_text_color(120, 120, 120)
                    pdf.cell(0, 5,
                             f"Реплик: {num_segments}  |  "
                             f"Общее время: {total_time:.1f} сек  |  "
                             f"Доля: {total_time / (audio_duration_min * 60) * 100:.0f}%",
                             ln=True)
                    pdf.ln(2)
            
            # Summary text
            pdf._use_font("", 11)
            pdf.set_text_color(44, 62, 80)
            
            if summary:
                pdf.multi_cell(0, 6, summary)
            else:
                pdf.set_text_color(180, 180, 180)
                pdf.multi_cell(0, 6, "(Суммаризация недоступна)")
            
            pdf.ln(6)
            
            # Light separator between speakers
            if idx < len(correction_df) - 1:
                pdf.set_draw_color(220, 220, 220)
                pdf.set_line_width(0.2)
                y = pdf.get_y()
                pdf.line(20, y, pdf.w - 20, y)
                pdf.ln(6)
    else:
        pdf._use_font("", 12)
        pdf.set_text_color(180, 100, 100)
        pdf.cell(0, 10, "Данные суммаризации отсутствуют", ln=True, align="C")
    
    # ---- Footer note ----
    pdf.ln(10)
    pdf.set_draw_color(189, 195, 199)
    pdf.set_line_width(0.3)
    y = pdf.get_y()
    pdf.line(10, y, pdf.w - 10, y)
    pdf.ln(4)
    
    pdf._use_font("", 8)
    pdf.set_text_color(170, 170, 170)
    pdf.cell(0, 5, f"Сгенерировано автоматически {now.strftime('%d.%m.%Y в %H:%M')}", ln=True, align="C")
    pdf.cell(0, 5, "Бот суммаризации переговоров", ln=True, align="C")
    
    # ---- Output to buffer ----
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    
    return buf
