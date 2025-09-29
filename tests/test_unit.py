# tests/test_unit.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import normalize_text

def test_normalize_text_numbers():
    """Testet die Umwandlung von Zahlen in Text."""
    assert normalize_text("Test 123 und 4 5 6") == "test einhundertdreiundzwanzig und vier fünf sechs"

def test_normalize_text_special_chars():
    """Testet die Entfernung von Satz- und Sonderzeichen."""
    assert normalize_text("Hallo, Welt! Wie geht's? (Toll)...") == "hallo welt wie gehts toll"

def test_normalize_text_umlauts_and_case():
    """Testet die korrekte Behandlung von Umlauten und Groß/Kleinschreibung."""
    assert normalize_text("Schöne GRÜSSE und süßes Gebäck") == "schöne grüsse und süßes gebäck"

def test_normalize_text_empty_and_whitespace():
    """Testet das Verhalten bei leeren oder nur aus Leerzeichen bestehenden Eingaben."""
    assert normalize_text("   ") == ""
    assert normalize_text("") == ""
    assert normalize_text(None) == ""

def test_normalize_double_space():
    """Testet die Korrektur von mehrfachen Leerzeichen."""
    assert normalize_text("Ein    Text   mit    vielen  Lücken") == "ein text mit vielen lücken"