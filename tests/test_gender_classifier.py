# tests/test_classifier.py

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Wir erstellen manuell Testdaten (Signal + Samplerate), um von echten Dateien unabhängig zu sein
sr = 16000

# Ein tiefer Ton, der klar in den männlichen Bereich fällt
y_male = np.sin(2 * np.pi * 120 * np.arange(sr) / sr, dtype=np.float32)

# Ein hoher Ton, der klar in den weiblichen Bereich fällt
y_female = np.sin(2 * np.pi * 220 * np.arange(sr) / sr, dtype=np.float32)

# Ein Ton im unklaren Bereich
y_unknown = np.sin(2 * np.pi * 200 * np.arange(sr) / sr, dtype=np.float32)

def test_classifier_predict_male_fallback(classifier_instance):
    """Testet, ob ein tiefer Ton über die Fallback-Heuristik als männlich erkannt wird."""
    # Wir testen hier bewusst die Heuristik (kein geladenes Modell)
    classifier_instance.model = None
    label, conf = classifier_instance.predict(y_male, sr)
    assert label == "männlich"

def test_classifier_predict_female_fallback(classifier_instance):
    """Testet, ob ein hoher Ton über die Fallback-Heuristik als weiblich erkannt wird."""
    classifier_instance.model = None
    label, conf = classifier_instance.predict(y_female, sr)
    assert label == "weiblich"

def test_classifier_predict_unknown_fallback(classifier_instance):
    """Testet, ob ein uneindeutiger Ton als 'unbekannt' klassifiziert wird."""
    classifier_instance.model = None
    label, conf = classifier_instance.predict(y_unknown, sr)
    assert label == "unbekannt"