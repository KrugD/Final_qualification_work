# Offline Meeting Transcription Pipeline

Система автоматического протоколирования офлайн переговоров с использованием нейросетевой обработки речи и больших языковых моделей.

## Описание

Проект реализует полный пайплайн обработки аудиозаписей совещаний и переговоров:

1. **Диаризация (Diarization)** — определение говорящих и временных меток их речи
2. **Распознавание речи (ASR)** — транскрибация аудио в текст
3. **Суммаризация (Summarization)** — создание краткого содержания для каждого спикера
4. **Коррекция текста (Correction)** — исправление ошибок в суммаризированном тексте
5. **Кластеризация спикеров (Speaker Clustering)** — объединение спикеров из разных частей длинных аудиофайлов

## Структура проекта

```
Final_qualification_work/
├── pipeline/                      # Основные модули пайплайна
│   ├── __init__.py               # Экспорты модуля
│   ├── pipeline.py               # Главный скрипт запуска пайплайна
│   ├── diarization.py            # Модуль диаризации спикеров
│   ├── asr.py                    # Модуль распознавания речи
│   ├── summarization.py          # Модуль суммаризации текста
│   ├── correction.py             # Модуль коррекции текста
│   └── speaker_clustering.py     # Модуль кластеризации спикеров
│
├── utils/                         # Вспомогательные модули
│   ├── __init__.py
│   ├── config.py                 # Конфигурация моделей и параметров
│   └── models.py                 # Функции загрузки моделей
│
├── audio_test/                    # Директория для входных аудиофайлов
│
├── pipeline_output/               # Директория с результатами обработки
│   └── <audio_name>/
│       ├── <audio_name>_diarization.txt
│       ├── <audio_name>_asr.txt
│       ├── <audio_name>_summarization.txt
│       ├── <audio_name>_correction.txt
│       ├── speaker_clusters_visualization.png  # Для длинных аудио
│       └── clustering_metrics_report.txt       # Для длинных аудио
│
├── run.py                         # Точка входа для запуска пайплайна
├── pyproject.toml                 # Зависимости проекта
├── uv.lock                        # Lock-файл зависимостей
├── .env                           # Переменные окружения (HF_TOKEN)
└── README.md
```

## Используемые модели

| Задача | Модель |
|--------|--------|
| Диаризация | `pyannote/speaker-diarization-3.1` |
| Распознавание речи | `openai/whisper-small` |
| Суммаризация | `RussianNLP/FRED-T5-Summarizer` |
| Коррекция текста | `ai-forever/sage-m2m100-1.2B` |
| Эмбеддинги спикеров | `pyannote/embedding` |

## Требования

- Python >= 3.10
- CUDA (рекомендуется для ускорения)
- HuggingFace токен с доступом к моделям pyannote

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd Final_qualification_work
```

### 2. Установка зависимостей

С использованием [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Или с использованием pip:

```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
HF_TOKEN=your_huggingface_token_here
```

Для получения токена:
1. Зарегистрируйтесь на [HuggingFace](https://huggingface.co/)
2. Примите условия использования модели [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Создайте токен в настройках аккаунта

## Использование

### Запуск через uv (рекомендуется)

Все команды выполняются из **корневой папки проекта**:

```bash
# Обработка конкретного аудиофайла
uv run python run.py --audio-file your_audio.wav

# Обработка всех файлов в директории audio_test/
uv run python run.py

# С принудительным анализом спикеров (визуализация + метрики)
uv run python run.py --audio-file your_audio.wav --force-clustering
```

### Запуск через python (без uv)

```bash
# Из корневой папки
python run.py --audio-file your_audio.wav

# Или перейдя в папку pipeline
cd pipeline
python pipeline.py --audio-file your_audio.wav
```

### Запуск отдельных модулей

Из папки `pipeline/`:

```bash
cd pipeline

# Только диаризация
python diarization.py

# Только распознавание речи (требуется файл диаризации)
python asr.py

# Только суммаризация (требуется файл ASR)
python summarization.py

# Только коррекция (требуется файл суммаризации)
python correction.py
```

## Выходные файлы

После обработки аудиофайла создаются следующие файлы:

| Файл | Описание |
|------|----------|
| `*_diarization.txt` | Результаты диаризации с временными метками |
| `*_asr.txt` | Транскрипция речи по спикерам |
| `*_summarization.txt` | Краткое содержание речи каждого спикера |
| `*_correction.txt` | Исправленные суммаризации |

### Для длинных аудиофайлов (> 50 минут) или с флагом `--force-clustering`

| Файл | Описание |
|------|----------|
| `speaker_clusters_visualization.png` | Визуализация кластеров спикеров (t-SNE, PCA) |
| `clustering_metrics_report.txt` | Метрики качества кластеризации |

> **Примечание:** Для коротких файлов используйте флаг `--force-clustering`, чтобы получить визуализацию и метрики.

## Метрики качества кластеризации

Система вычисляет следующие метрики (автоматически для длинных файлов или с `--force-clustering`):

- **Silhouette Score** — мера схожести точек с их кластером (от -1 до 1, выше лучше)
- **Calinski-Harabasz Index** — отношение межкластерной к внутрикластерной дисперсии
- **Davies-Bouldin Index** — средняя схожесть между кластерами (ниже лучше)

## Конфигурация

Параметры моделей настраиваются в файле `utils/config.py`:

```python
# Минимальная длительность сегмента речи
MIN_SEGMENT_DURATION = 0.5  # секунды

# Максимальная длина текста для суммаризации
MAX_SUMMARY_INPUT_LENGTH = 2000  # символы

# Порог расстояния для кластеризации спикеров
CLUSTERING_DISTANCE_THRESHOLD = 0.4

# Порог для определения длинного аудио
MAX_CHUNK_DURATION = 50  # минуты
```

## Лицензия

MIT License

## Автор

Выпускная квалификационная работа на тему: "Протоколирование офлайн переговоров с использованием нейросетевой обработки речи и больших языковых моделей"
