import os
import pickle
from typing import Dict, Tuple

import librosa
import numpy as np

from audio_processing import safe_load_audio
from config import GENDER_MODEL_PATH
from utils import gui_log

# Überprüfen, ob scikit-learn optional importiert werden kann
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

class GenderClassifier:
    """
    Klassifiziert das Geschlecht von Sprechern in Audiodateien.
    Kann ein logistisches Regressionsmodell trainieren und verwenden oder auf einen
    einfachen Pitch-basierten Fallback zurückgreifen.
    """
    def __init__(self, model_path: str = GENDER_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        if HAVE_SKLEARN and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e:
                gui_log(None, "WARN", f"Konnte Gender-Modell nicht laden: {e}")
                self.model = None

    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extrahiert Pitch (f0) und MFCCs als Features aus einem Audiosignal."""
        try:
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            # Nehmen den Median des Pitch, um Ausreißer zu reduzieren
            pitch = float(np.nanmedian(f0)) if np.any(np.isfinite(f0)) else 150.0
        except Exception:
            pitch = 150.0 # Fallback-Wert

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        return np.hstack(([pitch], mfccs))

    def predict(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Sagt das Geschlecht für ein gegebenes Audiosignal voraus."""
        feats = self._extract_features(y, sr).reshape(1, -1)

        # Vorhersage mit trainiertem Modell, falls vorhanden
        if self.model is not None and HAVE_SKLEARN:
            try:
                probs = self.model.predict_proba(feats)[0]
                classes = list(self.model.classes_)
                idx = int(np.argmax(probs))
                return classes[idx], float(probs[idx])
            except Exception as e:
                gui_log(None, "WARN", f"Fehler bei der Gender-Vorhersage mit Modell: {e}")

        # Fallback auf einfache Pitch-basierte Logik
        pitch = float(feats[0, 0])
        if pitch < 175:
            return "männlich", 0.7
        elif pitch > 185:
            return "weiblich", 0.7
        else:
            return "unbekannt", 0.3

    def train(self, file_label_pairs, cv=3) -> Dict:
        """Trainiert ein neues Klassifizierungsmodell."""
        if not HAVE_SKLEARN:
            raise RuntimeError("scikit-learn ist für das Training nicht installiert.")

        X, y = [], []
        for path, label in file_label_pairs:
            try:
                s, sr = safe_load_audio(path)
                feats = self._extract_features(s, sr)
                X.append(feats)
                y.append(label)
            except Exception as e:
                gui_log(None, "WARN", f"Kalibrierung: Konnte {path} nicht verarbeiten: {e}")

        if len(X) < 4 or len(set(y)) < 2:
            raise ValueError("Mindestens 2 Beispiele pro Klasse (männlich/weiblich) für das Training notwendig.")

        X_matrix = np.vstack(X)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
        clf.fit(X_matrix, y)

        # Evaluierung des Modells
        cv_scores = cross_val_score(clf, X_matrix, y, cv=min(cv, len(y)))
        report = classification_report(y, clf.predict(X_matrix), output_dict=True)

        # Speichern des trainierten Modells
        try:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(clf, f)
            self.model = clf
            gui_log(None, "INFO", f"Neues Gender-Modell wurde trainiert und unter {self.model_path} gespeichert.")
        except Exception as e:
            gui_log(None, "WARN", f"Konnte trainiertes Gender-Modell nicht speichern: {e}")

        return {"cv_scores": cv_scores.tolist(), "report": report}
