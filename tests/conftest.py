# tests/conftest.py

import pytest
import numpy as np
import os
import sys
from pydub import AudioSegment

# Füge das Hauptverzeichnis zum Pfad hinzu, damit wir 'main' importieren können
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import GenderClassifier, Transcriber

@pytest.fixture(scope="session")
def classifier_instance():
    """Stellt eine einmalige Instanz des GenderClassifiers für alle Tests bereit."""
    # Wir übergeben None als Pfad, da wir das Laden eines echten Modells im Test nicht wollen.
    return GenderClassifier(model_path=None)

@pytest.fixture
def mock_transcriber(mocker):
    """
    Ersetzt ("mockt") unseren echten Transcriber.
    Anstatt das langsame KI-Modell zu laden, gibt diese Attrappe immer einen festen Text zurück.
    Das macht den Test extrem schnell und vorhersagbar.
    """
    mock = mocker.MagicMock(spec=Transcriber)
    mock.transcribe.return_value = {"text": "dies ist ein test"}
    return mock

@pytest.fixture
def temp_wav_file(tmp_path):
    """
    Erstellt eine temporäre, gültige WAV-Datei für Tests.
    `tmp_path` ist ein eingebautes pytest-Fixture, das ein temporäres Verzeichnis erstellt und automatisch aufräumt.
    """
    sr = 16000
    duration = 3.0 # Gültige Länge für unsere Pipeline
    frequency = 440.0
    t = np.linspace(0., duration, int(sr * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)

    audio_segment = AudioSegment(
        data.astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    
    # Simuliere eine gültige Lautstärke
    audio_segment = audio_segment - 20 # Reduziere auf -20dBFS

    file_path = tmp_path / "test.wav"
    audio_segment.export(file_path, format="wav")
    return str(file_path)