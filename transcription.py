import json
import os
import threading
from typing import Dict, Optional

from config import GERMAN_INITIAL_PROMPT, TRANSCRIPT_CACHE_PATH

# Optionale Importe für Whisper-Engines
try:
    import whisper
    HAVE_OPENAI_WHISPER = True
except ImportError:
    HAVE_OPENAI_WHISPER = False

try:
    from faster_whisper import WhisperModel as FWWhisperModel
    HAVE_FASTER_WHISPER = True
except ImportError:
    HAVE_FASTER_WHISPER = False


class TranscriptCache:
    """
    Ein einfacher, Thread-sicherer Cache für Transkriptionen, um wiederholte
    API-Aufrufe für identische Audiosegmente zu vermeiden.
    """
    def __init__(self, path: str = TRANSCRIPT_CACHE_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.cache = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (IOError, json.JSONDecodeError):
                self.cache = {}

    def get(self, sha: str) -> Optional[str]:
        """Ruft einen Eintrag aus dem Cache ab."""
        return self.cache.get(sha)

    def set(self, sha: str, text: str):
        """Fügt einen Eintrag zum Cache hinzu und speichert ihn auf der Festplatte."""
        with self.lock:
            self.cache[sha] = text
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except IOError:
                # Fehler beim Speichern des Caches ignorieren
                pass


class Transcriber:
    """
    Wrapper-Klasse für verschiedene Whisper-Implementierungen (OpenAI Whisper, faster-whisper).
    """
    def __init__(self, engine_preference: str, model_name: str, device: str = "cpu", compute_type: str = "int8"):
        self.engine, self.model = engine_preference, None

        # Automatische Auswahl der besten verfügbaren Engine
        engine_to_use = "openai"
        if engine_preference == "auto":
            if HAVE_FASTER_WHISPER:
                engine_to_use = "faster"
            elif HAVE_OPENAI_WHISPER:
                engine_to_use = "openai"
            else:
                raise RuntimeError("Keine Whisper-Implementierung (weder 'whisper' noch 'faster-whisper') gefunden.")
        else:
            engine_to_use = engine_preference

        self.engine = engine_to_use

        # Laden des Modells
        if self.engine == "openai":
            if not HAVE_OPENAI_WHISPER:
                raise ImportError("OpenAI Whisper ist nicht installiert. Bitte `pip install openai-whisper` ausführen.")
            self.model = whisper.load_model(model_name)
        elif self.engine == "faster":
            if not HAVE_FASTER_WHISPER:
                raise ImportError("faster-whisper ist nicht installiert. Bitte `pip install faster-whisper` ausführen.")
            self.model = FWWhisperModel(model_name, device=device, compute_type=compute_type)
        else:
            raise ValueError(f"Unbekannte Whisper-Engine: {self.engine}")

    def transcribe(self, wav_path: str) -> Dict:
        """Transkribiert eine WAV-Datei."""
        if self.engine == "openai":
            res = self.model.transcribe(wav_path, language="de", initial_prompt=GERMAN_INITIAL_PROMPT)
            return {"text": res.get("text", "")}
        else: # faster-whisper
            segments, _ = self.model.transcribe(wav_path, language="de", initial_prompt=GERMAN_INITIAL_PROMPT)
            # faster-whisper gibt einen Generator zurück, wir müssen ihn konsumieren
            full_text = " ".join([s.text for s in segments])
            return {"text": full_text}
