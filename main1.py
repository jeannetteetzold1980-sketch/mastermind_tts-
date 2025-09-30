# main.py — TTS Toolkit v12.0 (Optimierte Pipeline: Gender → Segmentierung → Preprocessing → Transkription → Export)

import os
import sys
import time
import json
import re
import hashlib
import shutil
import threading
import queue
import traceback
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

# --- 1. Abhängigkeitsprüfung und Importe ---
_is_testing = 'pytest' in sys.modules

def show_error_and_exit(message: str, exit_now: bool = True):
    """Zeigt eine Fehlermeldung an und beendet das Programm kontrolliert."""
    if not _is_testing:
        sg.popup_error(message, title="Toolkit Fehler")
    else:
        print(f"TEST-ERROR: {message}")
    if exit_now:
        sys.exit(1)

try:
    import numpy as np
    import PySimpleGUI as sg
    import librosa
    import soundfile as sf
    from pydub import AudioSegment, silence
    from num2words import num2words
    from scipy.signal import butter, filtfilt
except ImportError as e:
    show_error_and_exit(f"FEHLER: Eine Kernbibliothek fehlt: {e}\n\nBitte 'pip install -r requirements.txt' ausführen.")

HAVE_FASTER_WHISPER, HAVE_OPENAI_WHISPER, HAVE_SKLEARN, pickle = False, False, False, None
try:
    import whisper
    HAVE_OPENAI_WHISPER = True
except ImportError: pass
try:
    from faster_whisper import WhisperModel as FWWhisperModel
    HAVE_FASTER_WHISPER = True
except ImportError: pass
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    HAVE_SKLEARN = True
except ImportError: pass
try:
    import pickle
except ImportError:
    show_error_and_exit("FEHLER: Die Standardbibliothek 'pickle' konnte nicht importiert werden.")

# --- 2. Konfiguration & globale Pfade ---
BASE_OUTPUT_DIR = "results"; os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP}.log")
GENDER_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "gender_model.pkl")
TRANSCRIPT_CACHE_PATH = os.path.join(BASE_OUTPUT_DIR, "transcript_cache.json")
GERMAN_INITIAL_PROMPT = "Dies ist eine Transkription auf Deutsch. Sie enthält Umlaute wie ä, ö, ü und auch das ß."
DEFAULT_PREPROCESS_THREADS = max(1, min(8, (os.cpu_count() or 4)))
MIN_SEGMENT_SEC, MAX_SEGMENT_SEC, MIN_DBFS = 2.5, 12.0, -40.0
WHISPER_MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v3"]

# --- 3. Hilfsfunktionen (Logging, Audio, Text) ---
def ts() -> str: return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def write_log_file(message: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(message + "\n")
    except Exception: pass
def gui_log(q: Optional[queue.Queue], level: str, message: str):
    line = f"[{ts()}] [{level.upper()}] {message}"; write_log_file(line)
    if q is not None: q.put(("log", line))
    else: print(line)
def safe_load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    try: data, file_sr = sf.read(path, always_2d=False, dtype='float32'); data = np.mean(data, axis=1) if data.ndim > 1 else data; data = librosa.resample(y=data, orig_sr=file_sr, target_sr=sr) if file_sr != sr else data; return data, sr
    except Exception: y, _ = librosa.load(path, sr=sr, mono=True); return y.astype(np.float32), sr
def segment_sha1(segment: AudioSegment) -> str: return hashlib.sha1(segment.raw_data).hexdigest()
def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    def number_to_words(match):
        try: return num2words(int(match.group(0)), lang='de')
        except Exception: return match.group(0)
    text = re.sub(r'\d+', number_to_words, text.strip()); text = text.lower(); text = re.sub(r'[^a-zäöüß\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- 4. Kern-Klassen ---
class TranscriptCache:
    def __init__(self, path: str = TRANSCRIPT_CACHE_PATH):
        self.path = path; self.lock = threading.Lock(); self.cache = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f: self.cache = json.load(f)
            except Exception: self.cache = {}
    def get(self, sha: str) -> Optional[str]: return self.cache.get(sha)
    def set(self, sha: str, text: str):
        with self.lock:
            self.cache[sha] = text
            try:
                with open(self.path, "w", encoding="utf-8") as f: json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception: pass

class GenderClassifier:
    def __init__(self, model_path: str = GENDER_MODEL_PATH):
        self.model_path = model_path; self.model = None
        if HAVE_SKLEARN and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f: self.model = pickle.load(f)
            except Exception: self.model = None
    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        try: f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')); pitch = float(np.mean(f0[~np.isnan(f0)])) if np.any(~np.isnan(f0)) else 150.0
        except Exception: pitch = 150.0
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1); return np.hstack(([pitch], mfccs))
    def predict(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        feats = self._extract_features(y, sr).reshape(1, -1)
        if self.model is not None and HAVE_SKLEARN:
            try:
                probs = self.model.predict_proba(feats)[0]; classes = list(self.model.classes_)
                idx = int(np.argmax(probs)); return classes[idx], float(probs[idx])
            except Exception: pass
        pitch = float(feats[0, 0])
        if pitch < 175: return "männlich", 0.7
        elif pitch > 185: return "weiblich", 0.7
        else: return "unbekannt", 0.3
    def train(self, file_label_pairs, cv=3) -> Dict:
        if not HAVE_SKLEARN: raise RuntimeError("scikit-learn nicht installiert.")
        X, y = [], []
        for path, label in file_label_pairs:
            try: s, sr = safe_load_audio(path); feats = self._extract_features(s, sr); X.append(feats); y.append(label)
            except Exception as e: gui_log(None, "WARN", f"Kalibrierung: Konnte {path} nicht verarbeiten: {e}")
        if len(X) < 4 or len(set(y)) < 2: raise ValueError("Mindestens 2 Beispiele pro Klasse (männlich/weiblich) notwendig.")
        X = np.vstack(X); clf = LogisticRegression(max_iter=2000, class_weight='balanced'); clf.fit(X, y)
        cv_scores = cross_val_score(clf, X, y, cv=min(cv, len(y)))
        report = classification_report(y, clf.predict(X), output_dict=True)
        try:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            with open(self.model_path, "wb") as f: pickle.dump(clf, f)
            self.model = clf
        except Exception as e: gui_log(None, "WARN", f"Konnte trainiertes Modell nicht speichern: {e}")
        return {"cv_scores": cv_scores.tolist(), "report": report}

class Transcriber:
    def __init__(self, engine_preference: str, model_name: str, device: str = "cpu", compute_type: str = "int8"):
        self.engine, self.model = engine_preference, None
        engine_to_use = "openai"
        if engine_preference == "auto":
            if HAVE_FASTER_WHISPER: engine_to_use = "faster"
            elif HAVE_OPENAI_WHISPER: engine_to_use = "openai"
            else: raise RuntimeError("Keine Whisper-Implementierung (weder 'whisper' noch 'faster-whisper') gefunden.")
        else: engine_to_use = engine_preference
        self.engine = engine_to_use
        if self.engine == "openai":
            self.model = whisper.load_model(model_name)
        elif self.engine == "faster":
            self.model = FWWhisperModel(model_name, device=device, compute_type=compute_type)
    def transcribe(self, wav_path: str) -> Dict:
        if self.engine == "openai":
            res = self.model.transcribe(wav_path, language="de", initial_prompt=GERMAN_INITIAL_PROMPT); return {"text": res.get("text", "")}
        else: segments, _ = self.model.transcribe(wav_path, language="de"); return {"text": " ".join([s.text for s in segments])}

# --- 5. Pipeline-Funktionen (NEUE REIHENFOLGE) ---
def segment_audio(filepath: str) -> Tuple[AudioSegment, int]:
    """Lädt Audio und gibt vollständiges AudioSegment zurück."""
    try:
        y, sr = safe_load_audio(filepath, sr=16000)
        pcm16 = (y * 32767).astype(np.int16).tobytes()
        audio = AudioSegment(data=pcm16, sample_width=2, frame_rate=sr, channels=1)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Audio-Laden fehlgeschlagen: {e}")

def split_into_segments(audio: AudioSegment, orig_file: str, temp_dir: str) -> List[Dict]:
    """
    Intelligente Segmentierung basierend auf linguistischen Einheiten.
    
    Strategie:
    1. Finde alle Stille-Pausen (potenzielle Splitpunkte)
    2. Klassifiziere Pausen nach Länge (kurz/mittel/lang)
    3. Merge Segmente intelligent basierend auf:
       - Satzgrenzen (lange Pausen)
       - Atemgruppen (mittlere Pausen)
       - Zeitlimits (MAX_SEGMENT_SEC)
    4. Vermeide Schnitte bei kurzen Pausen (innerhalb von Wörtern/Phrasen)
    """
    
    # === SCHRITT 1: Finde alle Stille-Pausen ===
    # Weniger aggressive Erkennung für mehr Kontrolle
    initial_segments = silence.split_on_silence(
        audio, 
        min_silence_len=300,      # 300ms minimale Stille
        silence_thresh=MIN_DBFS,
        keep_silence=150          # Behalte 150ms Stille an Rändern
    )
    
    if not initial_segments:
        initial_segments = [audio]
    
    # === SCHRITT 2: Analysiere Pausen zwischen Segmenten ===
    def get_pause_duration(audio_full: AudioSegment, segments: List[AudioSegment]) -> List[float]:
        """Berechnet Pausenlängen zwischen Segmenten."""
        pauses = []
        current_pos = 0
        
        for seg in segments:
            seg_start = audio_full.find(seg, current_pos)
            if seg_start > current_pos:
                pause_ms = seg_start - current_pos
                pauses.append(pause_ms)
            else:
                pauses.append(0)
            current_pos = seg_start + len(seg)
        
        return pauses
    
    # Klassifiziere Pausen
    pauses = get_pause_duration(audio, initial_segments)
    
    # Pause-Kategorien:
    # - Kurz (< 200ms): Innerhalb von Phrasen/Wörtern → IMMER MERGEN
    # - Mittel (200-500ms): Atemgruppen → Mergen wenn möglich
    # - Lang (> 500ms): Satzgrenzen → Bevorzugte Splitpunkte
    
    PAUSE_SHORT = 200   # ms
    PAUSE_MEDIUM = 500  # ms
    
    # === SCHRITT 3: Intelligentes Merging ===
    merged_segments = []
    current_merge = initial_segments[0] if initial_segments else audio
    current_duration = len(current_merge) / 1000.0
    
    for idx in range(1, len(initial_segments)):
        seg = initial_segments[idx]
        pause_before = pauses[idx] if idx < len(pauses) else 0
        seg_duration = len(seg) / 1000.0
        potential_duration = current_duration + seg_duration
        
        # Entscheidungslogik für Merge
        should_merge = False
        
        # Fall 1: Kurze Pause → IMMER mergen (Wort/Phrasen-Kontinuität)
        if pause_before < PAUSE_SHORT:
            should_merge = True
            merge_reason = "short_pause"
        
        # Fall 2: Mittlere Pause + noch Platz → Mergen (Atemgruppe)
        elif pause_before < PAUSE_MEDIUM and potential_duration < MAX_SEGMENT_SEC * 0.85:
            should_merge = True
            merge_reason = "breath_group"
        
        # Fall 3: Aktuelles Segment zu kurz → Mergen (Mindestlänge)
        elif current_duration < MIN_SEGMENT_SEC:
            should_merge = True
            merge_reason = "too_short"
        
        # Fall 4: Nächstes Segment sehr kurz → Mergen (vermeide Mini-Segmente)
        elif seg_duration < MIN_SEGMENT_SEC * 0.7:
            should_merge = True
            merge_reason = "next_too_short"
        
        # Fall 5: Lange Pause aber Zeitlimit überschritten → Nicht mergen
        elif potential_duration > MAX_SEGMENT_SEC:
            should_merge = False
            merge_reason = "time_limit"
        
        # Fall 6: Lange Pause → Satzgrenze, nicht mergen
        else:
            should_merge = False
            merge_reason = "sentence_boundary"
        
        if should_merge and potential_duration <= MAX_SEGMENT_SEC:
            # Merge: Füge Pause + Segment hinzu
            silence_gap = AudioSegment.silent(duration=int(pause_before))
            current_merge = current_merge + silence_gap + seg
            current_duration = len(current_merge) / 1000.0
        else:
            # Split: Speichere aktuelles Segment und starte neu
            if current_duration >= MIN_SEGMENT_SEC:
                merged_segments.append(current_merge)
            current_merge = seg
            current_duration = seg_duration
    
    # Füge letztes Segment hinzu
    if current_duration >= MIN_SEGMENT_SEC:
        merged_segments.append(current_merge)
    elif merged_segments:  # Zu kurz → an vorheriges anhängen
        merged_segments[-1] = merged_segments[-1] + AudioSegment.silent(duration=200) + current_merge
    else:
        merged_segments.append(current_merge)
    
    # === SCHRITT 4: Post-Processing Quality Check ===
    out_segs = []
    base = os.path.splitext(os.path.basename(orig_file))[0]
    
    for idx, seg in enumerate(merged_segments):
        dur = len(seg) / 1000.0
        dbfs = seg.dBFS if seg.duration_seconds > 0 else -100.0
        
        # Qualitäts-Check
        if MIN_SEGMENT_SEC <= dur <= MAX_SEGMENT_SEC and dbfs > (MIN_DBFS - 5):
            sha = segment_sha1(seg)
            fname = f"{base}_{idx+1:03d}_{sha[:8]}.wav"
            out_segs.append({
                "wav_path": os.path.join(temp_dir, fname),
                "segment": seg,
                "sha": sha,
                "orig_file": orig_file,
                "duration_sec": dur
            })
    
    return out_segs

def preprocess_segment(seg_dict: Dict) -> Dict:
    """
    Professionelles Audio-Preprocessing für TTS-Training.
    
    Schritte:
    1. Noise Reduction (Spectral Gating)
    2. Normalisierung (Peak & LUFS)
    3. High-Pass Filter (Entfernung von Rumpeln <80Hz)
    4. De-Essing (Reduktion von Zischlauten)
    5. Trimming (Stille am Anfang/Ende entfernen)
    6. Quality Check (SNR, Clipping Detection)
    """
    try:
        segment = seg_dict['segment']
        
        # Konvertiere zu numpy für Verarbeitung
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        samples = samples / 32768.0  # Normalisiere auf [-1, 1]
        sr = segment.frame_rate
        
        # === 1. NOISE REDUCTION (Spectral Gating) ===
        # Berechne Short-Time Fourier Transform
        stft = librosa.stft(samples, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Noise Profile aus leisesten 10% der Frames
        frame_power = np.mean(magnitude**2, axis=0)
        noise_threshold = np.percentile(frame_power, 10)
        noise_frames = frame_power < noise_threshold
        noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
        
        # Spectral Gating: Reduziere Frequenzen unter Noise-Profil
        magnitude_denoised = np.maximum(magnitude - 1.5 * noise_profile, 0.0)
        
        # Rekonstruiere Signal
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        samples = librosa.istft(stft_denoised, hop_length=512, length=len(samples))
        
        # === 2. HIGH-PASS FILTER (80Hz) ===
        # Entfernt Rumpeln und DC-Offset
        from scipy.signal import butter, filtfilt
        nyquist = sr / 2
        cutoff = 80 / nyquist
        b, a = butter(4, cutoff, btype='high')
        samples = filtfilt(b, a, samples)
        
        # === 3. DE-ESSING (6-10kHz Bereich dämpfen) ===
        # Reduziert scharfe Zischlaute (S, T, Z)
        stft = librosa.stft(samples, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Frequenzbänder für De-Essing
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        deess_mask = np.ones_like(freqs)
        deess_range = (freqs >= 6000) & (freqs <= 10000)
        deess_mask[deess_range] = 0.6  # Dämpfe um 40%
        
        magnitude = magnitude * deess_mask[:, np.newaxis]
        stft_deessed = magnitude * np.exp(1j * phase)
        samples = librosa.istft(stft_deessed, hop_length=512, length=len(samples))
        
        # === 4. NORMALISIERUNG ===
        # Peak Normalisierung auf -3dB (lässt Headroom für weitere Verarbeitung)
        peak = np.abs(samples).max()
        if peak > 0:
            target_peak = 10 ** (-3 / 20)  # -3dB in linear
            samples = samples * (target_peak / peak)
        
        # LUFS-basierte Lautstärke-Normalisierung (psychoakustisch korrekt)
        # Approximation: RMS-Normalisierung auf -20dB
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0:
            target_rms = 10 ** (-20 / 20)  # -20dB in linear
            samples = samples * (target_rms / rms)
        
        # === 5. TRIMMING (Stille entfernen) ===
        # Entfernt Stille unter -40dB am Anfang/Ende
        samples, trim_indices = librosa.effects.trim(
            samples, 
            top_db=40,
            frame_length=2048,
            hop_length=512
        )
        
        # === 6. QUALITY CHECKS ===
        quality_metrics = {}
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(samples**2)
        noise_power = np.mean((samples - np.convolve(samples, np.ones(256)/256, mode='same'))**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_metrics['snr_db'] = float(snr_db)
        
        # Clipping Detection
        clipping_samples = np.sum(np.abs(samples) > 0.99)
        clipping_percent = (clipping_samples / len(samples)) * 100
        quality_metrics['clipping_percent'] = float(clipping_percent)
        
        # Dynamic Range
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
        quality_metrics['dynamic_range_db'] = float(dynamic_range)
        
        # Zero Crossing Rate (Indikator für Stimmhaftigkeit)
        zcr = librosa.feature.zero_crossing_rate(samples)[0]
        quality_metrics['mean_zcr'] = float(np.mean(zcr))
        
        # === QUALITY GATE ===
        # Verwerfe Segment bei schlechter Qualität
        if snr_db < 10:  # SNR zu niedrig
            seg_dict['quality_rejected'] = True
            seg_dict['reject_reason'] = f"Low SNR: {snr_db:.1f}dB"
            return seg_dict
        
        if clipping_percent > 0.5:  # Mehr als 0.5% Clipping
            seg_dict['quality_rejected'] = True
            seg_dict['reject_reason'] = f"Clipping: {clipping_percent:.2f}%"
            return seg_dict
        
        if len(samples) < sr * 1.0:  # Zu kurz nach Trimming (< 1 Sekunde)
            seg_dict['quality_rejected'] = True
            seg_dict['reject_reason'] = "Too short after trimming"
            return seg_dict
        
        # === KONVERTIERE ZURÜCK ZU AUDIOSEGMENT ===
        samples_int16 = (samples * 32767).astype(np.int16)
        processed_segment = AudioSegment(
            data=samples_int16.tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1
        )
        
        seg_dict['segment'] = processed_segment
        seg_dict['quality_metrics'] = quality_metrics
        seg_dict['quality_rejected'] = False
        
        return seg_dict
        
    except Exception as e:
        seg_dict['quality_rejected'] = True
        seg_dict['reject_reason'] = f"Processing error: {str(e)}"
        return seg_dict

# --- 6. Haupt-Pipeline-Worker ---
def pipeline_worker(file_list, model_engine, model_name, threads, gender_filter, include_unknown, q, stop_event, pause_event, classifier):
    tmp_session, out_session = None, None
    try:
        gui_log(q, "INFO", f"Pipeline startet mit neuer Reihenfolge...")
        session_id = TIMESTAMP
        tmp_session = os.path.join(BASE_OUTPUT_DIR, f"tmp_{session_id}")
        os.makedirs(tmp_session, exist_ok=True)
        
        # === SCHRITT 1: GENDER-FILTERUNG ===
        filtered_files = []
        if gender_filter != 'alle':
            gui_log(q, "INFO", f"🎯 SCHRITT 1: Gender-Filterung (nur '{gender_filter}' Dateien)...")
            for i, orig_file in enumerate(file_list):
                if stop_event.is_set(): raise InterruptedError()
                try:
                    y, sr = safe_load_audio(orig_file)
                    detected_gender, confidence = classifier.predict(y, sr)
                    
                    if detected_gender == gender_filter:
                        filtered_files.append(orig_file)
                        gui_log(q, "INFO", f"✓ {os.path.basename(orig_file)}: {detected_gender} ({confidence:.2f}) - AKZEPTIERT")
                    else:
                        gui_log(q, "INFO", f"✗ {os.path.basename(orig_file)}: {detected_gender} ({confidence:.2f}) - ÜBERSPRUNGEN")
                        
                except Exception as e:
                    gui_log(q, "WARN", f"Gender-Analyse fehlgeschlagen für {os.path.basename(orig_file)}: {e}")
                
                q.put(("progress_update", "1. Gender-Filter", i + 1, len(file_list)))
            
            if not filtered_files:
                raise ValueError(f"Keine Dateien entsprechen dem Filter '{gender_filter}'!")
            
            gui_log(q, "INFO", f"✓ Gender-Filterung abgeschlossen: {len(filtered_files)}/{len(file_list)} Dateien ausgewählt")
        else:
            filtered_files = file_list
            gui_log(q, "INFO", "Kein Gender-Filter aktiv - alle Dateien werden verarbeitet")
        
        # === SCHRITT 2: SEGMENTIERUNG ===
        gui_log(q, "INFO", f"📊 SCHRITT 2: Intelligente Audio-Segmentierung (linguistische Einheiten)...")
        all_segments = []
        segmentation_stats = {
            'total_files': len(filtered_files),
            'total_segments': 0,
            'avg_segments_per_file': 0,
            'avg_duration': 0
        }
        
        for i, filepath in enumerate(filtered_files):
            if stop_event.is_set(): raise InterruptedError()
            try:
                audio, sr = segment_audio(filepath)
                segments = split_into_segments(audio, filepath, tmp_session)
                all_segments.extend(segments)
                
                seg_durations = [s.get('duration_sec', 0) for s in segments]
                avg_dur = np.mean(seg_durations) if seg_durations else 0
                
                gui_log(q, "INFO", 
                       f"✓ {os.path.basename(filepath)}: {len(segments)} Segmente "
                       f"(Ø {avg_dur:.1f}s, Range: {min(seg_durations):.1f}-{max(seg_durations):.1f}s)")
                
            except Exception as e:
                gui_log(q, "WARN", f"Segmentierung fehlgeschlagen für {os.path.basename(filepath)}: {e}")
            
            q.put(("progress_update", "2. Segmentierung", i + 1, len(filtered_files)))
        
        if not all_segments:
            raise ValueError("Keine verwendbaren Segmente nach Segmentierung gefunden!")
        
        # Statistiken berechnen
        segmentation_stats['total_segments'] = len(all_segments)
        segmentation_stats['avg_segments_per_file'] = len(all_segments) / len(filtered_files)
        all_durations = [s.get('duration_sec', 0) for s in all_segments]
        segmentation_stats['avg_duration'] = float(np.mean(all_durations))
        
        gui_log(q, "INFO", 
               f"✓ Segmentierung abgeschlossen: {len(all_segments)} Segmente "
               f"(Ø {segmentation_stats['avg_segments_per_file']:.1f} pro Datei, "
               f"Ø Dauer: {segmentation_stats['avg_duration']:.1f}s)")
        
        # === SCHRITT 3: PREPROCESSING ===
        gui_log(q, "INFO", f"⚙️ SCHRITT 3: Professionelles Audio-Preprocessing...")
        processed_segments = []
        rejected_count = 0
        quality_stats = {
            'low_snr': 0,
            'clipping': 0,
            'too_short': 0,
            'processing_error': 0
        }
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_seg = {executor.submit(preprocess_segment, seg): seg for seg in all_segments}
            for i, future in enumerate(as_completed(future_to_seg)):
                if stop_event.is_set(): raise InterruptedError()
                try:
                    result = future.result()
                    
                    if result.get('quality_rejected', False):
                        rejected_count += 1
                        reason = result.get('reject_reason', 'unknown')
                        
                        # Kategorisiere Ablehnungsgründe
                        if 'SNR' in reason:
                            quality_stats['low_snr'] += 1
                        elif 'Clipping' in reason:
                            quality_stats['clipping'] += 1
                        elif 'short' in reason:
                            quality_stats['too_short'] += 1
                        else:
                            quality_stats['processing_error'] += 1
                        
                        gui_log(q, "WARN", f"Segment verworfen: {reason}")
                    else:
                        processed_segments.append(result)
                        metrics = result.get('quality_metrics', {})
                        if metrics:
                            gui_log(q, "DEBUG", 
                                   f"Segment OK - SNR: {metrics.get('snr_db', 0):.1f}dB, "
                                   f"DR: {metrics.get('dynamic_range_db', 0):.1f}dB")
                
                except Exception as e:
                    gui_log(q, "WARN", f"Preprocessing fehlgeschlagen für Segment: {e}")
                    rejected_count += 1
                    quality_stats['processing_error'] += 1
                
                q.put(("progress_update", "3. Preprocessing", i + 1, len(all_segments)))
        
        gui_log(q, "INFO", 
               f"✓ Preprocessing abgeschlossen: {len(processed_segments)} akzeptiert, "
               f"{rejected_count} verworfen")
        gui_log(q, "INFO", 
               f"Ablehnungsgründe: Low SNR={quality_stats['low_snr']}, "
               f"Clipping={quality_stats['clipping']}, "
               f"Too Short={quality_stats['too_short']}, "
               f"Errors={quality_stats['processing_error']}")
        
        if not processed_segments:
            raise ValueError("Keine Segmente haben die Quality-Checks bestanden!")
        
        # === SCHRITT 4: TRANSKRIPTION ===
        gui_log(q, "INFO", f"✍️ SCHRITT 4: Transkription mit Whisper...")
        transcriber = Transcriber(engine_preference=model_engine, model_name=model_name)
        gui_log(q, "INFO", f"Transcriber-Engine: {transcriber.engine}")
        cache = TranscriptCache()
        
        meta_data = []
        for i, seg in enumerate(processed_segments):
            if stop_event.is_set(): break
            while pause_event.is_set(): 
                time.sleep(0.1)
                continue
            
            cached = cache.get(seg['sha'])
            if cached:
                seg['transcript'] = cached
                gui_log(q, "INFO", f"Cache-Hit für Segment {i+1}")
            else:
                seg_path = seg['wav_path']
                seg['segment'].export(seg_path, format="wav")
                res = transcriber.transcribe(seg_path)
                seg['transcript'] = normalize_text(res.get('text', ''))
                
                if seg['transcript']:
                    cache.set(seg['sha'], seg['transcript'])
                
                try: 
                    os.remove(seg_path)
                except OSError: 
                    pass
            
            if seg.get('transcript'):
                meta_data.append(seg)
            
            q.put(("progress_update", "4. Transkription", i + 1, len(processed_segments)))
        
        if stop_event.is_set(): raise InterruptedError()
        if not meta_data: 
            raise ValueError("Keine nutzbaren Segmente nach Transkription übrig.")
        
        gui_log(q, "INFO", f"✓ Transkription abgeschlossen: {len(meta_data)} transkribierte Segmente")
        
        # === SCHRITT 5: EXPORT ===
        gui_log(q, "INFO", f"📦 SCHRITT 5: Export & ZIP-Erstellung...")
        out_session = os.path.join(BASE_OUTPUT_DIR, f"session_{session_id}")
        wavs_out = os.path.join(out_session, "wavs")
        os.makedirs(wavs_out, exist_ok=True)
        meta_path = os.path.join(out_session, "metadata.tsv")
        
        with open(meta_path, "w", encoding="utf-8-sig", newline="") as mf:
            for item in meta_data:
                wavname = os.path.basename(item["wav_path"])
                item['segment'].export(os.path.join(wavs_out, wavname), format="wav")
                mf.write(f"{wavname}\t{item['transcript']}\n")
        
        zip_path = shutil.make_archive(out_session, "zip", out_session)
        shutil.rmtree(out_session)
        
        gui_log(q, "INFO", f"✓ Export abgeschlossen: {zip_path}")
        q.put(("processing_complete", len(file_list), len(meta_data), len(all_segments) - len(meta_data), zip_path))
        
    except InterruptedError:
        gui_log(q, "INFO", "Prozess gestoppt.")
        q.put(('stopped',))
    except Exception:
        q.put(('error', traceback.format_exc()))
    finally:
        if tmp_session and os.path.exists(tmp_session):
            shutil.rmtree(tmp_session, ignore_errors=True)
        if (stop_event and stop_event.is_set() and out_session and os.path.exists(out_session)):
            shutil.rmtree(out_session, ignore_errors=True)

# --- 7. GUI-Funktionen ---
def create_main_window():
    sg.theme("DarkGrey13")
    left_col = [
        [sg.Text("Zu verarbeitende Dateien:")],
        [sg.Listbox(values=[], key="-FILES-", size=(60, 12), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, enable_events=True)],
        [sg.Button("Hinzufügen", key="-ADD-"), sg.Button("Entfernen", key="-REMOVE-"), sg.Button("Alle löschen", key="-CLEAR-")],
        [sg.Text("Drag & Drop Dateien/Ordner hierher", font=("Helvetica", 8), key="-DROP_TEXT-")]
    ]
    right_col = [
        [sg.Text("Einstellungen", font=("Helvetica", 10, "bold"))],
        [sg.Text("Whisper Modell:"), sg.Combo(WHISPER_MODEL_OPTIONS, default_value="small", key="-MODEL-")],
        [sg.Text("Engine:"), sg.Combo(["auto", "faster", "openai"], default_value="auto", key="-ENGINE-")],
        [sg.Text("Preproc. Threads:"), sg.Spin([i for i in range(1, 9)], initial_value=DEFAULT_PREPROCESS_THREADS, key="-THREADS-", size=(3,1))],
        [sg.Frame('Geschlechter-Filter (Schritt 1)', [
            [sg.Radio('Alle', 'G', k='-G_A-', default=True)],
            [sg.Radio('Männlich', 'G', k='-G_M-')],
            [sg.Radio('Weiblich', 'G', k='-G_F-')]
        ])],
        [sg.Button("Ressourcen laden", key="-LOAD-"), sg.Button("Start", key="-START-", disabled=True)],
        [sg.Button("🎙️ Gender-Kalibrierung", key="-CALIB-"), sg.Button("Log speichern", key="-SAVELOG-")]
    ]
    layout = [
        [sg.Text("TTS Datensatz Formatter v12.0", font=("Helvetica", 16))],
        [sg.Column(left_col), sg.VSeperator(), sg.Column(right_col)],
        [sg.Frame("Fortschritt", [
            [sg.Text("Bereit.", size=(80,1), key='-STATUS-')],
            [sg.ProgressBar(100, key="-PROG_ALL-", size=(80, 20))]
        ])],
        [sg.Frame("Aktionen", [
            [sg.Button('Pause', key='-PAUSE-', visible=False), sg.Button('Stopp', key='-STOP-', visible=False)]
        ], element_justification='center', key='-ACTION_FRAME-')],
        [sg.Frame("Logs", [
            [sg.Multiline("", size=(120, 16), key="-LOG-", autoscroll=True, disabled=True)]
        ])]
    ]
    return sg.Window("TTS Toolkit v12.0 (Optimierte Pipeline)", layout, finalize=True)
    
def calibration_dialog(classifier: GenderClassifier):
    layout = [
        [sg.Text("Gender-Kalibrierung")],
        [sg.FilesBrowse("Dateien auswählen", key="-CAL_FILES-"), sg.Button("Laden", key="-CAL_LOAD-")],
        [sg.Table(values=[], headings=["Datei", "Label"], key="-CAL_TABLE-", auto_size_columns=False, col_widths=[80, 12], num_rows=10, select_mode=sg.TABLE_SELECT_MODE_EXTENDED, enable_events=True)],
        [sg.Button("Als Männlich", key="-CAL_M-"), sg.Button("Als Weiblich", key="-CAL_F-")],
        [sg.Button("Training starten", key="-CAL_TRAIN-"), sg.Button("Schließen", key="-CAL_CLOSE-")],
        [sg.Text("", key="-CAL_STATUS-")]
    ]
    win = sg.Window("Gender-Kalibrierung", layout, modal=True, finalize=True)
    files, labels = [], []
    
    def refresh_table():
        rows = [[os.path.basename(files[i]), labels[i]] for i in range(len(files))]
        win["-CAL_TABLE-"].update(values=rows)
        m, f, u = labels.count("männlich"), labels.count("weiblich"), labels.count("unlabeled")
        win["-CAL_STATUS-"].update(f"Markiert: m={m}, w={f}, unmarkiert={u}")
    
    while True:
        event, values = win.read()
        if event in (sg.WIN_CLOSED, "-CAL_CLOSE-"): 
            break
        
        if event == "-CAL_LOAD-":
            raw = values.get("-CAL_FILES-", "")
            files = [p for p in raw.split(";") if p and os.path.isfile(p)] if raw else []
            if files:
                labels = ["unlabeled"] * len(files)
                refresh_table()
        
        elif event in ("-CAL_M-", "-CAL_F-"):
            sel = values.get("-CAL_TABLE-")
            if sel:
                lab = "männlich" if event == "-CAL_M-" else "weiblich"
                for idx in sel:
                    labels[idx] = lab
                refresh_table()
        
        elif event == "-CAL_TRAIN-":
            if not HAVE_SKLEARN:
                sg.popup_error("scikit-learn ist nicht installiert.")
                continue
            
            pairs = [(files[i], labels[i]) for i in range(len(files)) if labels[i] != "unlabeled"]
            if sum(1 for _,l in pairs if l=="männlich") < 2 or sum(1 for _,l in pairs if l=="weiblich") < 2:
                sg.popup_error("Bitte min. 2 Beispiele pro Geschlecht markieren.")
                continue
            
            try:
                res = classifier.train(pairs)
                cv_scores, report = res.get("cv_scores", []), res.get("report", {})
                txt = f"Training abgeschlossen.\nCV Accuracy: {np.mean(cv_scores):.3f}\n\nReport:\n"
                for cls, metrics in report.items():
                    if isinstance(metrics, dict):
                        txt += f"{cls}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}\n"
                sg.popup_scrolled(txt, title="Trainingsergebnis")
                break
            except Exception as e:
                sg.popup_error(f"Fehler: {e}")
    
    win.close()
    
def main():
    window = create_main_window()
    window['-FILES-'].bind('<Drop>', '+DRAG_DROP')
    window['-DROP_TEXT-'].bind('<Drop>', '+DRAG_DROP')
    
    classifier, transcriber_obj = GenderClassifier(), None
    file_set = set()
    worker_thread, update_q, stop_event, pause_event = None, None, None, None
    model_loaded = False
    
    def update_start_button():
        window["-START-"].update(disabled=(not file_set or not model_loaded))
    
    def add_files_to_list(paths_to_add):
        for path in paths_to_add:
            path = path.strip()
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if not file.startswith('.'):
                            file_set.add(os.path.join(root, file))
            elif os.path.isfile(path):
                if not os.path.basename(path).startswith('.'):
                    file_set.add(path)
        window['-FILES-'].update(sorted(list(file_set)))
        update_start_button()
    
    def load_transcriber_worker(q, engine, model_name):
        nonlocal transcriber_obj, model_loaded
        try:
            gui_log(q, "INFO", f"Lade Transcriber-Modell '{model_name}' mit Engine '{engine}'...")
            transcriber_obj = Transcriber(engine, model_name)
            model_loaded = True
            q.put(("model_loaded", True))
        except Exception:
            q.put(("error", traceback.format_exc()))

    while True:
        event, values = window.read(timeout=100)
        
        if update_q:
            try:
                while True:
                    msg_type, *payload = update_q.get_nowait()
                    
                    if msg_type == 'log':
                        window['-LOG-'].update(payload[0] + "\n", append=True)
                    
                    elif msg_type == 'progress_update':
                        stage, current, total = payload
                        window['-STATUS-'].update(f'Phase: {stage}... ({current} von {total})')
                        window['-PROG_ALL-'].update(current, total)
                    
                    elif msg_type in ('stopped', 'error', 'processing_complete'):
                        if msg_type == 'processing_complete':
                            gui_log(update_q, "DONE", f"Prozess abgeschlossen. Zip: {payload[3]}")
                        elif msg_type == 'error':
                            gui_log(update_q, "ERROR", payload[0])
                        
                        window['-STATUS-'].update("Bereit.")
                        window['-PROG_ALL-'].update(0, 1)
                        window['-START-'].update(disabled=False)
                        window['-PAUSE-'].update(visible=False)
                        window['-STOP-'].update(visible=False)
                        window['-ACTION_FRAME-'].update(visible=False)
                        worker_thread = None
                    
                    elif msg_type == 'model_loaded':
                        model_loaded = True
                        update_start_button()
                        gui_log(update_q, "INFO", "Ressourcen bereit.")
            
            except queue.Empty:
                pass
        
        if event == sg.WIN_CLOSED:
            if worker_thread and worker_thread.is_alive():
                if stop_event:
                    stop_event.set()
                worker_thread.join(timeout=2)
            break

        if event == '-ADD-':
            paths_str = sg.popup_get_file(
                "Wähle Dateien",
                multiple_files=True,
                file_types=(("Audio Files", "*.*"), ("All Files", "*.*"))
            )
            if paths_str:
                add_files_to_list(paths_str.split(";"))
        
        elif event.endswith('+DRAG_DROP'):
            key = event.split('+')[0]
            add_files_to_list(values[key])

        elif event == '-REMOVE-':
            for s in values['-FILES-']:
                file_set.discard(s)
            window['-FILES-'].update(sorted(list(file_set)))
            update_start_button()
            
        elif event == '-CLEAR-':
            file_set.clear()
            window['-FILES-'].update([])
            update_start_button()

        elif event == '-LOAD-':
            model_loaded = False
            update_start_button()
            update_q = queue.Queue()
            gui_log(update_q, "INFO", "Lade Ressourcen...")
            threading.Thread(
                target=load_transcriber_worker,
                args=(update_q, values["-ENGINE-"], values["-MODEL-"]),
                daemon=True
            ).start()
            
        elif event == '-START-':
            update_q = queue.Queue()
            stop_event, pause_event = threading.Event(), threading.Event()
            
            window['-START-'].update(disabled=True)
            window['-ACTION_FRAME-'].update(visible=True)
            window['-PAUSE-'].update(visible=True, disabled=False, text="Pause")
            window['-STOP-'].update(visible=True, disabled=False)
            
            # Gender-Filter bestimmen
            if values['-G_M-']:
                gender_choice = 'männlich'
            elif values['-G_F-']:
                gender_choice = 'weiblich'
            else:
                gender_choice = 'alle'
            
            worker_thread = threading.Thread(
                target=pipeline_worker,
                args=(
                    sorted(list(file_set)),
                    values["-ENGINE-"],
                    values["-MODEL-"],
                    int(values["-THREADS-"]),
                    gender_choice,
                    True,
                    update_q,
                    stop_event,
                    pause_event,
                    classifier
                ),
                daemon=True
            )
            worker_thread.start()
            
        elif event == '-PAUSE-':
            if pause_event:
                if pause_event.is_set():
                    pause_event.clear()
                    window['-PAUSE-'].update("Pause")
                else:
                    pause_event.set()
                    window['-PAUSE-'].update("Fortsetzen")
                
        elif event == '-STOP-':
            if stop_event:
                stop_event.set()
                window['-STOP-'].update(disabled=True)
                window['-PAUSE-'].update(disabled=True)
            
        elif event == '-CALIB-':
            if HAVE_SKLEARN:
                calibration_dialog(classifier)
            else:
                sg.popup_error("scikit-learn ist nicht installiert. Kalibrierung nicht möglich.")
            
        elif event == '-SAVELOG-':
            path = sg.popup_get_file(
                "Log speichern",
                save_as=True,
                default_path=f"log_{TIMESTAMP}.txt",
                file_types=(("Log Datei", "*.log"),)
            )
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(window['-LOG-'].get())
                    sg.popup("Log gespeichert!")
                except Exception as e:
                    sg.popup_error(f"Fehler: {e}")
    
    window.close()
    
if __name__ == "__main__":
    main()