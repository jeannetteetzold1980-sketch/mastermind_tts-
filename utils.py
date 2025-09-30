import datetime
import queue
from typing import Optional

# Globale Variable für den Log-Dateipfad, wird zur Laufzeit gesetzt
LOG_FILE = None

def set_log_file(path: str):
    """Setzt den globalen Pfad für die Log-Datei."""
    global LOG_FILE
    LOG_FILE = path

def ts() -> str:
    """Gibt einen Zeitstempel-String zurück."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log_file(message: str):
    """Schreibt eine Nachricht in die globale Log-Datei."""
    if not LOG_FILE:
        return
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        # Fehler beim Loggen ignorieren, um die Hauptanwendung nicht zu stören
        pass

def gui_log(q: Optional[queue.Queue], level: str, message: str):
    """Formatiert eine Log-Nachricht, schreibt sie in die Log-Datei und sendet sie an die GUI-Queue."""
    line = f"[{ts()}] [{level.upper()}] {message}"
    write_log_file(line)
    if q is not None:
        q.put(("log", line))
    else:
        # Fallback, wenn keine GUI-Queue vorhanden ist (z.B. für Tests)
        print(line)
