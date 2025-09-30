# main.py — TTS Toolkit v11.0 (Finale Master-Version)

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
# Diese Sektion stellt sicher, dass alle nötigen Bibliotheken vorhanden sind,
# bevor die GUI überhaupt versucht zu starten.

# Umschalte-Flag, um Popups beim automatisierten Testen zu unterdrücken
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

# --- 5. Pipeline & Worker ---
def preprocess_file(filepath: str, temp_dir: str) -> List[Dict]:
    try: y, sr = safe_load_audio(filepath, sr=16000)
    except Exception as e: return [{"error": f"Ladefehler: {e}", "orig_file": filepath}]
    try: pcm16 = (y * 32767).astype(np.int16).tobytes()
    except Exception as e: return [{"error": f"PCM-Konvertierungsfehler: {e}", "orig_file": filepath}]
    audio = AudioSegment(data=pcm16, sample_width=2, frame_rate=sr, channels=1)
    segments = silence.split_on_silence(audio, min_silence_len=700, silence_thresh=MIN_DBFS, keep_silence=250)
    if not segments: segments = [audio]
    out_segs = []; base = os.path.splitext(os.path.basename(filepath))[0]
    for idx, seg in enumerate(segments):
        dur = len(seg) / 1000.0
        dbfs = seg.dBFS if seg.duration_seconds > 0 else -100.0
        if MIN_SEGMENT_SEC < dur < MAX_SEGMENT_SEC and dbfs is not None and dbfs > (MIN_DBFS - 5):
            sha = segment_sha1(seg); fname = f"{base}_{idx+1:03d}_{sha[:8]}.wav"
            out_segs.append({"wav_path": os.path.join(temp_dir, fname), "segment": seg, "sha": sha, "orig_file": filepath})
    return out_segs

def pipeline_worker(file_list, model_engine, model_name, threads, gender_filter, include_unknown, q, stop_event, pause_event, classifier):
    tmp_session, out_session = None, None
    try:
        gui_log(q, "INFO", f"Pipeline startet..."); session_id = TIMESTAMP; tmp_session = os.path.join(BASE_OUTPUT_DIR, f"tmp_{session_id}"); os.makedirs(tmp_session, exist_ok=True)
        transcriber = Transcriber(engine_preference=model_engine, model_name=model_name); gui_log(q, "INFO", f"Transcriber-Engine: {transcriber.engine}"); cache = TranscriptCache()
        all_segments = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_file = {executor.submit(preprocess_file, fp, tmp_session): fp for fp in file_list}
            for i, future in enumerate(as_completed(future_to_file)):
                if stop_event.is_set(): raise InterruptedError()
                all_segments.extend(future.result()); q.put(("progress_update", "Preprocessing", i + 1, len(file_list)))
        if stop_event.is_set(): raise InterruptedError()
        gui_log(q, "INFO", f"Preprocessing fertig, {len(all_segments)} Segmente gefunden.")
        usable_segments = []
        if gender_filter != 'alle':
            gui_log(q, "INFO", "Führe Gender-Filterung durch..."); original_files = sorted(list(set(seg['orig_file'] for seg in all_segments if 'orig_file' in seg)))
            file_gender_map = {}
            for i, orig_file in enumerate(original_files):
                if stop_event.is_set(): raise InterruptedError()
                y, sr = safe_load_audio(orig_file); file_gender_map[orig_file], _ = classifier.predict(y, sr); q.put(("progress_update", "Gender-Analyse", i + 1, len(original_files)))
            for seg in all_segments:
                if 'orig_file' in seg and file_gender_map.get(seg['orig_file']) == gender_filter: usable_segments.append(seg)
            gui_log(q, "INFO", f"{len(usable_segments)} Segmente nach Filterung übrig.")
        else: usable_segments = all_segments
        meta_data = []
        for i, seg in enumerate(usable_segments):
            if stop_event.is_set(): break
            while pause_event.is_set(): time.sleep(0.1); continue
            cached = cache.get(seg['sha'])
            if cached: seg['transcript'] = cached
            else:
                seg_path = seg['wav_path']; seg['segment'].export(seg_path, format="wav"); res = transcriber.transcribe(seg_path); seg['transcript'] = normalize_text(res.get('text', ''))
                if seg['transcript']: cache.set(seg['sha'], seg['transcript'])
                try: os.remove(seg_path)
                except OSError: pass
            if seg.get('transcript'): meta_data.append(seg)
            q.put(("progress_update", "Transkription", i + 1, len(usable_segments)))
        if stop_event.is_set(): raise InterruptedError()
        if not meta_data: raise ValueError("Keine nutzbaren Segmente nach Transkription übrig.")
        out_session = os.path.join(BASE_OUTPUT_DIR, f"session_{session_id}"); wavs_out = os.path.join(out_session, "wavs"); os.makedirs(wavs_out, exist_ok=True); meta_path = os.path.join(out_session, "metadata.tsv")
        with open(meta_path, "w", encoding="utf-8-sig", newline="") as mf:
            for item in meta_data:
                wavname = os.path.basename(item["wav_path"]); item['segment'].export(os.path.join(wavs_out, wavname), format="wav"); mf.write(f"{wavname}\t{item['transcript']}\n")
        zip_path = shutil.make_archive(out_session, "zip", out_session); shutil.rmtree(out_session)
        q.put(("processing_complete", len(file_list), len(meta_data), len(all_segments) - len(meta_data), zip_path))
    except InterruptedError: gui_log(q, "INFO", "Prozess gestoppt."); q.put(('stopped',))
    except Exception: q.put(('error', traceback.format_exc()))
    finally:
        if tmp_session and os.path.exists(tmp_session): shutil.rmtree(tmp_session, ignore_errors=True)
        if (stop_event and stop_event.is_set() and out_session and os.path.exists(out_session)): shutil.rmtree(out_session, ignore_errors=True)

# --- 6. GUI-Funktionen ---
def create_main_window():
    sg.theme("DarkGrey13")
    left_col = [[sg.Text("Zu verarbeitende Dateien:")], [sg.Listbox(values=[], key="-FILES-", size=(60, 12), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, enable_events=True)], [sg.Button("Hinzufügen", key="-ADD-"), sg.Button("Entfernen", key="-REMOVE-"), sg.Button("Alle löschen", key="-CLEAR-")], [sg.Text("Drag & Drop Dateien/Ordner hierher", font=("Helvetica", 8), key="-DROP_TEXT-")]]
    right_col = [[sg.Text("Einstellungen", font=("Helvetica", 10, "bold"))], [sg.Text("Whisper Modell:"), sg.Combo(WHISPER_MODEL_OPTIONS, default_value="small", key="-MODEL-")], [sg.Text("Engine:"), sg.Combo(["auto", "faster", "openai"], default_value="auto", key="-ENGINE-")], [sg.Text("Preproc. Threads:"), sg.Spin([i for i in range(1, 9)], initial_value=DEFAULT_PREPROCESS_THREADS, key="-THREADS-", size=(3,1))], [sg.Frame('Geschlechter-Filter', [[sg.Radio('Alle', 'G', k='-G_A-', default=True)], [sg.Radio('Männlich', 'G', k='-G_M-')], [sg.Radio('Weiblich', 'G', k='-G_F-')]])],[sg.Button("Ressourcen laden", key="-LOAD-"), sg.Button("Start", key="-START-", disabled=True)],[sg.Button("🎙️ Gender-Kalibrierung", key="-CALIB-"), sg.Button("Log speichern", key="-SAVELOG-")]]
    layout = [[sg.Text("TTS Datensatz Formatter", font=("Helvetica", 16))], [sg.Column(left_col), sg.VSeperator(), sg.Column(right_col)], [sg.Frame("Fortschritt", [[sg.Text("Bereit.", size=(80,1), key='-STATUS-')], [sg.ProgressBar(100, key="-PROG_ALL-", size=(80, 20))]])], [sg.Frame("Aktionen", [[sg.Button('Pause', key='-PAUSE-', visible=False), sg.Button('Stopp', key='-STOP-', visible=False)]], element_justification='center', key='-ACTION_FRAME-')], [sg.Frame("Logs", [[sg.Multiline("", size=(120, 16), key="-LOG-", autoscroll=True, disabled=True)]])]]
    return sg.Window("TTS Toolkit v11.0 (Final)", layout, finalize=True)
    
def calibration_dialog(classifier: GenderClassifier):
    layout = [[sg.Text("Gender-Kalibrierung")], [sg.FilesBrowse("Dateien auswählen", key="-CAL_FILES-"), sg.Button("Laden", key="-CAL_LOAD-")], [sg.Table(values=[], headings=["Datei", "Label"], key="-CAL_TABLE-", auto_size_columns=False, col_widths=[80, 12], num_rows=10, select_mode=sg.TABLE_SELECT_MODE_EXTENDED, enable_events=True)], [sg.Button("Als Männlich", key="-CAL_M-"), sg.Button("Als Weiblich", key="-CAL_F-")], [sg.Button("Training starten", key="-CAL_TRAIN-"), sg.Button("Schließen", key="-CAL_CLOSE-")], [sg.Text("", key="-CAL_STATUS-")]]
    win = sg.Window("Gender-Kalibrierung", layout, modal=True, finalize=True)
    files, labels = [], []
    def refresh_table():
        rows = [[os.path.basename(files[i]), labels[i]] for i in range(len(files))]; win["-CAL_TABLE-"].update(values=rows); m,f,u = labels.count("männlich"), labels.count("weiblich"), labels.count("unlabeled"); win["-CAL_STATUS-"].update(f"Markiert: m={m}, w={f}, unmarkiert={u}")
    while True:
        event, values = win.read();
        if event in (sg.WIN_CLOSED, "-CAL_CLOSE-"): break
        if event == "-CAL_LOAD-":
            raw = values.get("-CAL_FILES-", ""); files = [p for p in raw.split(";") if p and os.path.isfile(p)] if raw else []
            if files: labels = ["unlabeled"] * len(files); refresh_table()
        elif event in ("-CAL_M-", "-CAL_F-"):
            sel = values.get("-CAL_TABLE-");
            if sel: lab = "männlich" if event == "-CAL_M-" else "weiblich"; [labels.__setitem__(idx, lab) for idx in sel]; refresh_table()
        elif event == "-CAL_TRAIN-":
            if not HAVE_SKLEARN: sg.popup_error("scikit-learn ist nicht installiert."); continue
            pairs = [(files[i], labels[i]) for i in range(len(files)) if labels[i] != "unlabeled"]
            if sum(1 for _,l in pairs if l=="männlich") < 2 or sum(1 for _,l in pairs if l=="weiblich") < 2: sg.popup_error("Bitte min. 2 Beispiele pro Geschlecht markieren."); continue
            try:
                res = classifier.train(pairs); cv_scores, report = res.get("cv_scores", []), res.get("report", {}); txt = f"Training abgeschlossen.\nCV Accuracy: {np.mean(cv_scores):.3f}\n\nReport:\n"
                for cls, metrics in report.items():
                    if isinstance(metrics, dict): txt += f"{cls}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}\n"
                sg.popup_scrolled(txt, title="Trainingsergebnis"); break
            except Exception as e: sg.popup_error(f"Fehler: {e}")
    win.close()
    
def main():
    window = create_main_window()
    window['-FILES-'].bind('<Drop>', '+DRAG_DROP')
    window['-DROP_TEXT-'].bind('<Drop>', '+DRAG_DROP')
    
    classifier, transcriber_obj = GenderClassifier(), None
    file_set = set(); worker_thread, update_q, stop_event, pause_event = None, None, None, None
    model_loaded = False
    
    def update_start_button(): window["-START-"].update(disabled=(not file_set or not model_loaded))
    def add_files_to_list(paths_to_add):
        for path in paths_to_add:
            path = path.strip()
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files: 
                        if not file.startswith('.'): file_set.add(os.path.join(root, file))
            elif os.path.isfile(path):
                if not os.path.basename(path).startswith('.'): file_set.add(path)
        window['-FILES-'].update(sorted(list(file_set))); update_start_button()
    def load_transcriber_worker(q, engine, model_name):
        nonlocal transcriber_obj, model_loaded
        try:
            gui_log(q, "INFO", f"Lade Transcriber-Modell '{model_name}' mit Engine '{engine}'...")
            transcriber_obj = Transcriber(engine, model_name)
            model_loaded=True
            q.put(("model_loaded", True))
        except Exception:
            q.put(("error", traceback.format_exc()))

    while True:
        event, values = window.read(timeout=100)
        
        if update_q:
            try:
                while True:
                    msg_type, *payload = update_q.get_nowait()
                    if msg_type == 'log': window['-LOG-'].update(payload[0] + "\n", append=True)
                    elif msg_type == 'progress_update':
                        stage, current, total = payload
                        window['-STATUS-'].update(f'Phase: {stage}... ({current} von {total})')
                        window['-PROG_ALL-'].update(current, total)
                    elif msg_type in ('stopped', 'error', 'processing_complete'):
                        if msg_type == 'processing_complete': gui_log(update_q, "DONE", f"Prozess abgeschlossen. Zip: {payload[3]}")
                        elif msg_type == 'error': gui_log(update_q, "ERROR", payload[0])
                        window['-STATUS-'].update("Bereit."); window['-PROG_ALL-'].update(0, 1)
                        window['-START-'].update(disabled=False); window['-PAUSE-'].update(visible=False); window['-STOP-'].update(visible=False)
                        window['-ACTION_FRAME-'].update(visible=False); worker_thread = None
                    elif msg_type == 'model_loaded': model_loaded = True; update_start_button(); gui_log(update_q, "INFO", "Ressourcen bereit.")
            except queue.Empty: pass
        
        if event == sg.WIN_CLOSED:
            if worker_thread and worker_thread.is_alive():
                if stop_event: stop_event.set()
                worker_thread.join(timeout=2)
            break

        if event == '-ADD-':
            paths_str = sg.popup_get_file("Wähle Dateien", multiple_files=True, file_types=(("Audio Files", "*.*"), ("All Files", "*.*")));
            if paths_str: add_files_to_list(paths_str.split(";"))
        
        # Kombiniertes Drag & Drop Event Handling
        elif event.endswith('+DRAG_DROP'):
            key = event.split('+')[0]
            add_files_to_list(values[key])

        elif event == '-REMOVE-':
            for s in values['-FILES-']: file_set.discard(s)
            window['-FILES-'].update(sorted(list(file_set))); update_start_button()
            
        elif event == '-CLEAR-':
            file_set.clear(); window['-FILES-'].update([]); update_start_button()

        elif event == '-LOAD-':
            model_loaded = False; update_start_button(); update_q = queue.Queue(); gui_log(update_q, "INFO", "Lade Ressourcen...")
            threading.Thread(target=load_transcriber_worker, args=(update_q, values["-ENGINE-"], values["-MODEL-"]), daemon=True).start()
            
        elif event == '-START-':
            update_q = queue.Queue(); stop_event, pause_event = threading.Event(), threading.Event()
            window['-START-'].update(disabled=True); window['-ACTION_FRAME-'].update(visible=True); window['-PAUSE-'].update(visible=True, disabled=False, text="Pause"); window['-STOP-'].update(visible=True, disabled=False)
            worker_thread = threading.Thread(target=pipeline_worker, args=(sorted(list(file_set)), values["-ENGINE-"], values["-MODEL-"], int(values["-THREADS-"]), 'männlich' if values['-G_M-'] else 'weiblich' if values['-G_F-'] else 'alle', True, update_q, stop_event, pause_event, classifier), daemon=True)
            worker_thread.start()
            
        elif event == '-PAUSE-':
            if pause_event:
                if pause_event.is_set(): pause_event.clear(); window['-PAUSE-'].update("Pause")
                else: pause_event.set(); window['-PAUSE-'].update("Fortsetzen")
                
        elif event == '-STOP-':
            if stop_event: stop_event.set(); window['-STOP-'].update(disabled=True); window['-PAUSE-'].update(disabled=True)
            
        elif event == '-CALIB-':
            if HAVE_SKLEARN: calibration_dialog(classifier)
            else: sg.popup_error("scikit-learn ist nicht installiert. Kalibrierung nicht möglich.")
            
        elif event == '-SAVELOG-':
            path = sg.popup_get_file("Log speichern", save_as=True, default_path=f"log_{TIMESTAMP}.txt", file_types=(("Log Datei", "*.log"),))
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f: f.write(window['-LOG-'].get())
                    sg.popup("Log gespeichert!")
                except Exception as e: sg.popup_error(f"Fehler: {e}")
    
    window.close()
    
if __name__ == "__main__":
    main()