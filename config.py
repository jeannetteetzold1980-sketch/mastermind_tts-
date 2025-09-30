import os
from typing import List

# Verzeichnis für alle Ausgabe-Dateien
BASE_OUTPUT_DIR = "results"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Pfade für Modelle und Caches
GENDER_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "gender_model.pkl")
TRANSCRIPT_CACHE_PATH = os.path.join(BASE_OUTPUT_DIR, "transcript_cache.json")

# --- Whisper & Transkription ---
GERMAN_INITIAL_PROMPT: str = "Dies ist eine Transkription auf Deutsch. Sie enthält Umlaute wie ä, ö, ü und auch das ß."
WHISPER_MODEL_OPTIONS: List[str] = ["tiny", "base", "small", "medium", "large-v3"]

# --- Pipeline-Parameter ---
DEFAULT_PREPROCESS_THREADS: int = max(1, min(8, (os.cpu_count() or 4)))
MIN_SEGMENT_SEC: float = 2.5
MAX_SEGMENT_SEC: float = 12.0
MIN_DBFS: float = -40.0
