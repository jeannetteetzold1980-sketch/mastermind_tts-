import sys
import traceback

# Überprüfen, ob alle Abhängigkeiten vorhanden sind, bevor die GUI importiert wird,
# um verständlichere Fehlermeldungen bei fehlenden Paketen zu geben.
try:
    import numpy
    import PySimpleGUI as sg
    import librosa
    import soundfile
    import pydub
    import num2words
    import scipy
except ImportError as e:
    print(f"FEHLER: Eine Kernbibliothek fehlt: {e}", file=sys.stderr)
    print("Bitte führen Sie 'pip install -r requirements.txt' aus, um alle Abhängigkeiten zu installieren.", file=sys.stderr)
    sys.exit(1)

# Importiere die GUI-Funktionen erst nach der Überprüfung der Abhängigkeiten
from gui import start_gui, show_error_and_exit

if __name__ == "__main__":
    try:
        start_gui()
    except Exception as e:
        # Fange alle unerwarteten Fehler ab und zeige sie in einem Popup an
        tb = traceback.format_exc()
        show_error_and_exit(f"Ein unerwarteter Fehler ist aufgetreten:\n\n{tb}")
