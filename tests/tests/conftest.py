# tests/conftest.py
import pytest
import os

# Definiere die Pfade zu den Test-Audiodateien
@pytest.fixture(scope="session")
def male_audio_path():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_male.wav'))
    if not os.path.exists(path):
        pytest.fail(f"Testdatei nicht gefunden: {path}")
    return path

@pytest.fixture(scope="session")
def female_audio_path():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_female.wav'))
    if not os.path.exists(path):
        pytest.fail(f"Testdatei nicht gefunden: {path}")
    return path