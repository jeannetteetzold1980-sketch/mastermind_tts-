# main.py — TTS Toolkit v10.1 (Final, Testable, mit allen Fixes)
# Diese Version integriert alle professionellen Features und ist für die Test-Suite vorbereitet.

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

import numpy as np
import PySimpleGUI as sg

# --- Abhängigkeitsprüfung und Importe ---
try:
    import librosa
except ImportError as e:
    sg.popup_error(f"FEHLER: librosa wird benötigt, konnte aber nicht importiert werden:\n{e}")
    sys.exit(1)

HAVE_SOUNDFILE = True
try:
    import soundfile as sf
except ImportError:
    HAVE_SOUNDFILE = False

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError as e:
    sg.popup_error(f"FEHLER: pydub wird benötigt, konnte aber nicht importiert werden:\n{e}")
    sys.exit(1)

HAVE_OPENAI_WHISPER = True
try:
    import whisper
except ImportError:
    HAVE_OPENAI_WHISPER = False

HAVE_FASTER_WHISPER = True
try:
    from faster_whisper import WhisperModel as FWWhisperModel
except ImportError:
    HAVE_FASTER_WHISPER = False

try:
    from num2words import num2words
except ImportError as e:
    sg.popup_error(f"FEHLER: num2words wird benötigt, konnte aber nicht importiert werden:\n{e}")
    sys.exit(1)

HAVE_SKLEARN = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
except ImportError:
    HAVE_SKLEARN = False
    
try:
    import pickle
except ImportError as e:
     sg.popup_error(f"FEHLER: pickle wird benötigt, konnte aber nicht importiert werden:\n{e}")
     sys.exit(1)

# --- Konfiguration & globale Pfade ---
BASE_OUTPUT_DIR = "results"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP}.log")
GENDER_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "gender_model.pkl")
TRANSCRIPT_CACHE_PATH = os.path.join(BASE_OUTPUT_DIR, "transcript_cache.json")
GERMAN_INITIAL_PROMPT = "Dies ist eine Transkription auf Deutsch. Sie enthält Umlaute wie ä, ö, ü und auch das ß."

DEFAULT_PREPROCESS_THREADS = max(1, min(6, (os.cpu_count() or 4)))
MIN_SEGMENT_SEC = 2.5
MAX_SEGMENT_SEC = 12.0
MIN_DBFS = -40.0
WHISPER_MODEL_OPTIONS = ["small", "medium", "large-v2"]

# --- Hilfsfunktionen (Logging, Audio, Text) ---
def ts() -> str: return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log_file(message: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(message + "\n")
    except Exception: pass

def gui_log(q: Optional[queue.Queue], level: str, message: str):
    line = f"[{ts()}] [{level.upper()}] {message}"
    write_log_file(line)
    if q is not None: q.put(("log", line))
    else: print(line)

def safe_load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    if HAVE_SOUNDFILE:
        try:
            data, file_sr = sf.read(path, always_2d=False, dtype='float32')
            if data.ndim > 1: data = np.mean(data, axis=1)
            if file_sr != sr: data = librosa.resample(y=data, orig_sr=file_sr, target_sr=sr)
            return data, sr
        except Exception: pass
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), file_sr

def segment_sha1(segment: AudioSegment) -> str: return hashlib.sha1(segment.raw_data).hexdigest()

def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    def number_to_words(match):
        try: return num2words(int(match.group(0)), lang='de')
        except Exception: return match.group(0)
    text = re.sub(r'\d+', number_to_words, text.strip())
    text = text.lower()
    text = re.sub(r'[^a-zäöüß\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- Kern-Klassen (Transcriber, Cache, Classifier) ---
class TranscriptCache:
    def __init__(self, path: str = TRANSCRIPT_CACHE_PATH):
        self.path = path; self.lock = threading.Lock(); self.cache = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f: self.cache = json.load(f)
            except Exception: self.cache = {}
    def get(self, sha: str) -> Optional[str]:
        with self.lock: return self.cache.get(sha)
    def set(self, sha: str, text: str):
        with self.lock:
            self.cache[sha] = text
            try:
                with open(self.path, "w", encoding="utf-8") as f: json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception: pass

class GenderClassifier:
    def __init__(self, model_path: Optional[str] = GENDER_MODEL_PATH):
        self.model_path = model_path; self.model = None
        if self.model_path and HAVE_SKLEARN and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f: self.model = pickle.load(f)
            except Exception: self.model = None
    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            valid = f0[~np.isnan(f0)]
            pitch = float(np.mean(valid)) if valid.size > 0 else 150.0
        except Exception: pitch = 150.0
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        return np.hstack(([pitch], mfccs))
    def predict(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        feats = self._extract_features(y, sr).reshape(1, -1)
        if self.model is not None:
            try:
                probs = self.model.predict_proba(feats)[0]; classes = list(self.model.classes_)
                idx = int(np.argmax(probs)); label = classes[idx]; conf = float(probs[idx])
                return label, conf
            except Exception: pass
        pitch = float(feats[0, 0])
        if pitch < 200: return "männlich", 0.7
        elif pitch > 280: return "weiblich", 0.7
        else: return "unbekannt", 0.3
    def train(self, file_label_pairs: List[Tuple[str, str]], cv: int = 3) -> Dict:
        if not HAVE_SKLEARN: raise RuntimeError("scikit-learn nicht installiert.")
        X, y = [], []
        for path, label in file_label_pairs:
            try:
                s, sr = safe_load_audio(path, sr=16000); feats = self._extract_features(s, sr)
                X.append(feats); y.append(label)
            except Exception as e: gui_log(None, "WARN", f"Kalibrierung: konnte {path} nicht verarbeiten: {e}")
        if len(X) < 4 or len(set(y)) < 2: raise ValueError("Mindestens 2 Beispiele pro Klasse notwendig (insgesamt min 4).")
        X = np.vstack(X); clf = LogisticRegression(max_iter=2000); clf.fit(X, y)
        cv_scores = cross_val_score(clf, X, y, cv=min(cv, len(y)))
        report = classification_report(y, clf.predict(X), output_dict=True)
        try:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            with open(self.model_path, "wb") as f: pickle.dump(clf, f)
            self.model = clf
        except Exception as e: gui_log(None, "WARN", f"Konnte Modell nicht speichern: {e}")
        return {"cv_scores": cv_scores.tolist(), "report": report}

class Transcriber:
    def __init__(self, engine_preference: str, model_name: str, device: str = "cpu", compute_type: str = "int8"):
        self.engine, self.model = engine_preference, None
        if self.engine == "openai":
            if not HAVE_OPENAI_WHISPER: raise RuntimeError("OpenAI whisper ist nicht installiert.")
            self.model = whisper.load_model(model_name)
        elif self.engine == "faster":
            if not HAVE_FASTER_WHISPER: raise RuntimeError("faster-whisper ist nicht installiert.")
            self.model = FWWhisperModel(model_name, device=device, compute_type=compute_type)
        else: # auto
            if HAVE_FASTER_WHISPER:
                try: self.model = FWWhisperModel(model_name, device=device, compute_type=compute_type); self.engine = "faster"
                except Exception: self.engine = "openai" # fallback
            else: self.engine = "openai"
            
            if self.engine == "openai":
                if not HAVE_OPENAI_WHISPER: raise RuntimeError("Keine Whisper-Implementierung gefunden.")
                self.model = whisper.load_model(model_name)

    def transcribe(self, wav_path: str) -> Dict:
        if self.engine == "openai":
            res = self.model.transcribe(wav_path, language="de", initial_prompt=GERMAN_INITIAL_PROMPT)
            return {"text": res.get("text", "")}
        else: # faster-whisper
            segments, _ = self.model.transcribe(wav_path, language="de")
            return {"text": " ".join([s.text for s in segments])}

# --- Pipeline Worker ---
def preprocess_file(filepath: str, temp_dir: str) -> List[Dict]:
    try: y, sr = safe_load_audio(filepath, sr=16000)
    except Exception as e: return [{"error": f"Ladefehler: {e}"}]
    try: pcm16 = (y * 32767).astype(np.int16).tobytes()
    except Exception as e: return [{"error": f"PCM-Konvertierungsfehler: {e}"}]
    audio = AudioSegment(data=pcm16, sample_width=2, frame_rate=sr, channels=1)
    segments = split_on_silence(audio, min_silence_len=700, silence_thresh=MIN_DBFS, keep_silence=250)
    if not segments: segments = [audio]
    out_segs = []
    for idx, seg in enumerate(segments):
        dur = len(seg) / 1000.0
        if MIN_SEGMENT_SEC < dur < MAX_SEGMENT_SEC and seg.dBFS and seg.dBFS > (MIN_DBFS - 5):
            sha = segment_sha1(seg)
            fname = f"{os.path.splitext(os.path.basename(filepath))[0]}_{idx+1:03d}_{sha[:8]}.wav"
            out_segs.append({"wav_path": os.path.join(temp_dir, fname), "segment": seg, "sha": sha, "orig_file": filepath})
    return out_segs

def pipeline_worker(
    file_list: List[str], model_engine: str, model_name: str, preprocess_threads: int, gender_filter: str, include_unknown: bool,
    update_q: queue.Queue, stop_event: threading.Event, pause_event: threading.Event, classifier: GenderClassifier,
    _test_transcriber_obj=None):

    tmp_session, out_session = None, None
    try:
        if stop_event.is_set():
            raise InterruptedError()
        gui_log(update_q, "INFO", f"Pipeline startet...")
        session_id = TIMESTAMP
        tmp_session = os.path.join(BASE_OUTPUT_DIR, f"tmp_{session_id}")
        os.makedirs(tmp_session, exist_ok=True)
        
        transcriber = _test_transcriber_obj if _test_transcriber_obj else Transcriber(engine_preference=model_engine, model_name=model_name)
        if not _test_transcriber_obj:
            gui_log(update_q, "INFO", f"Transcriber-Engine: {transcriber.engine}")
        
        cache = TranscriptCache()
        
        all_segments = []
        with ThreadPoolExecutor(max_workers=preprocess_threads) as executor:
            future_to_file = {executor.submit(preprocess_file, fp, tmp_session): fp for fp in file_list}
            for i, future in enumerate(as_completed(future_to_file)):
                if stop_event.is_set(): raise InterruptedError()
                all_segments.extend(future.result())
                update_q.put(("preprocess_progress", i + 1, len(file_list)))
        
        gui_log(update_q, "INFO", f"Preprocessing fertig, {len(all_segments)} Segmente gefunden.")

        usable_segments = []
        if gender_filter != 'alle':
            gui_log(update_q, "INFO", "Führe Gender-Filterung durch...")
            file_gender_map = {}
            for seg in all_segments:
                orig = seg['orig_file']
                if orig not in file_gender_map:
                    y, sr = safe_load_audio(orig); file_gender_map[orig], _ = classifier.predict(y, sr)
                if file_gender_map[orig] == gender_filter:
                    usable_segments.append(seg)
            gui_log(update_q, "INFO", f"{len(usable_segments)} Segmente nach Filterung übrig.")
        else:
            usable_segments = all_segments
        
        transcribed_count = 0; meta_data = []
        for seg in usable_segments:
            if stop_event.is_set(): raise InterruptedError()
            while pause_event.is_set(): stop_event.wait(0.1)
            cached = cache.get(seg['sha'])
            if cached: seg['transcript'] = cached
            else:
                seg_path = seg['wav_path']
                seg['segment'].export(seg_path, format="wav")
                res = transcriber.transcribe(seg_path)
                seg['transcript'] = normalize_text(res.get('text', ''))
                if seg['transcript']: cache.set(seg['sha'], seg['transcript'])
                try: os.remove(seg_path)
                except OSError: pass

            if seg.get('transcript'): meta_data.append(seg)
            transcribed_count += 1
            update_q.put(("transcription_progress", transcribed_count, len(usable_segments)))
        
        if not meta_data: raise ValueError("Keine nutzbaren Segmente nach Transkription übrig.")
        
        out_session = os.path.join(BASE_OUTPUT_DIR, f"session_{session_id}")
        wavs_out = os.path.join(out_session, "wavs"); os.makedirs(wavs_out, exist_ok=True)
        meta_path = os.path.join(out_session, "metadata.tsv")
        with open(meta_path, "w", encoding="utf-8") as mf:
            for item in meta_data:
                wavname = os.path.basename(item["wav_path"])
                item['segment'].export(os.path.join(wavs_out, wavname), format="wav")
                mf.write(f"{wavname}\t{item['transcript']}\n")
        
        zip_path = shutil.make_archive(out_session, "zip", out_session)
        shutil.rmtree(out_session)
        update_q.put(("processing_complete", len(file_list), len(meta_data), len(all_segments) - len(meta_data), zip_path))

    except InterruptedError:
        gui_log(update_q, "INFO", "Prozess gestoppt.")
        update_q.put(('stopped',))
    except Exception as e:
        gui_log(update_q, "ERROR", f"Fehler in Pipeline: {traceback.format_exc()}")
        update_q.put(('error', str(e)))
    finally:
        if tmp_session and os.path.exists(tmp_session): shutil.rmtree(tmp_session, ignore_errors=True)
        if (stop_event and stop_event.is_set() and out_session and os.path.exists(out_session)): 
            shutil.rmtree(out_session, ignore_errors=True)

# --- GUI Erstellung ---
def create_main_window():
    sg.theme("DarkGrey13")
    left_col = [[sg.Text("Zu verarbeitende Dateien:")], [sg.Listbox(values=[], key="-FILES-", size=(60, 12), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, enable_events=True)], [sg.Button("Hinzufügen", key="-ADD-"), sg.Button("Entfernen", key="-REMOVE-"), sg.Button("Alle löschen", key="-CLEAR-")], [sg.Text("Drag & Drop Dateien/Ordner hierher", font=("Helvetica", 8))]]
    right_col = [[sg.Text("Einstellungen", font=("Helvetica", 10, "bold"))], [sg.Text("Whisper Modell:"), sg.Combo(WHISPER_MODEL_OPTIONS, default_value="small", key="-MODEL-")], [sg.Text("Engine:"), sg.Combo(["auto", "faster", "openai"], default_value="auto", key="-ENGINE-")], [sg.Text("Preproc. Threads:"), sg.Spin([i for i in range(1, 9)], initial_value=DEFAULT_PREPROCESS_THREADS, key="-THREADS-", size=(3,1))], [sg.Frame('Geschlechter-Filter', [[sg.Radio('Alle', 'G', k='-G_A-', default=True)], [sg.Radio('Männlich', 'G', k='-G_M-')], [sg.Radio('Weiblich', 'G', k='-G_F-')]])],[sg.Button("Ressourcen laden", key="-LOAD-"), sg.Button("Start", key="-START-", disabled=True)],[sg.Button("🎙️ Gender-Kalibrierung", key="-CALIB-"), sg.Button("Log speichern", key="-SAVELOG-")]]
    layout = [[sg.Text("TTS Datensatz Formatter", font=("Helvetica", 16))], [sg.Column(left_col), sg.VSeperator(), sg.Column(right_col)], [sg.Frame("Fortschritt", [[sg.Text("", size=(80,1), key='-STATUS-')], [sg.ProgressBar(100, key="-PROG_ALL-", size=(80, 20))]])], [sg.Frame("Aktionen", [[sg.Button('Pause', key='-PAUSE-', visible=False), sg.Button('Stopp', key='-STOP-', visible=False)]], element_justification='center', key='-ACTION_FRAME-')], [sg.Frame("Logs", [[sg.Multiline("", size=(120, 16), key="-LOG-", autoscroll=True, disabled=True)]])]]
    return sg.Window("TTS Toolkit v10.1", layout, finalize=True)
    
def calibration_dialog(classifier: GenderClassifier):
    # (Unverändert von Version 10)
    layout = [[sg.Text("Gender-Kalibrierung")], [sg.FilesBrowse("Dateien auswählen", key="-CAL_FILES-", multiple_files=True), sg.Button("Laden", key="-CAL_LOAD-")], [sg.Table(values=[], headings=["Datei", "Label"], key="-CAL_TABLE-", auto_size_columns=False, col_widths=[80, 12], num_rows=10, select_mode=sg.TABLE_SELECT_MODE_EXTENDED)], [sg.Button("Als Männlich", key="-CAL_M-"), sg.Button("Als Weiblich", key="-CAL_F-")], [sg.Button("Training starten", key="-CAL_TRAIN-"), sg.Button("Schließen", key="-CAL_CLOSE-")], [sg.Text("", key="-CAL_STATUS-")]]
    win = sg.Window("Gender-Kalibrierung", layout, modal=True, finalize=True)
    files, labels = [], []
    def refresh_table():
        rows = [[os.path.basename(files[i]), labels[i]] for i in range(len(files))]
        win["-CAL_TABLE-"].update(values=rows); m,f,u = labels.count("männlich"), labels.count("weiblich"), labels.count("unlabeled"); win["-CAL_STATUS-"].update(f"Markiert: m={m}, w={f}, unmarkiert={u}")
    while True:
        event, values = win.read();
        if event in (sg.WIN_CLOSED, "-CAL_CLOSE-"): break
        if event == "-CAL_LOAD-": raw = values.get("-CAL_FILES-", ""); files = [p for p in raw.split(";") if p]; labels = ["unlabeled"] * len(files); refresh_table()
        elif event in ("-CAL_M-", "-CAL_F-"):
            sel = values.get("-CAL_TABLE-");
            if sel: lab = "männlich" if event == "-CAL_M-" else "weiblich"; [labels.__setitem__(i, lab) for i in sel]; refresh_table()
        elif event == "-CAL_TRAIN-":
            pairs = [(files[i], labels[i]) for i in range(len(files)) if labels[i] != "unlabeled"]
            if sum(1 for _,l in pairs if l=="männlich") < 2 or sum(1 for _,l in pairs if l=="weiblich") < 2: sg.popup_error("Bitte min. 2 Beispiele pro Geschlecht markieren."); continue
            try:
                res = classifier.train(pairs)
                cv_scores, report = res.get("cv_scores", []), res.get("report", {}); txt = f"Training abgeschlossen.\nCV Accuracy: {np.mean(cv_scores):.3f}\n\nReport:\n"
                for cls, metrics in report.items():
                    if isinstance(metrics, dict): txt += f"{cls}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}\n"
                sg.popup_scrolled(txt, title="Trainingsergebnis"); break
            except Exception as e: sg.popup_error(f"Fehler: {e}")
    win.close()
    
def main():
    window = create_main_window()
    classifier, transcriber_obj = GenderClassifier(), None
    file_set = set(); worker_thread, update_q, stop_event, pause_event = None, None, None, None
    model_loaded = False
    
    def update_start_button(): window["-START-"].update(disabled=(not file_set or not model_loaded))

    def add_files_to_list(paths_to_add):
        for path in paths_to_add:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files: file_set.add(os.path.join(root, file))
            elif os.path.isfile(path):
                file_set.add(path)
        window['-FILES-'].update(sorted(list(file_set)))
        update_start_button()

    def load_transcriber_worker(q, engine, model_name):
        try:
            nonlocal transcriber_obj
            transcriber_obj = Transcriber(engine, model_name)
            q.put(("model_loaded", True))
        except Exception as e:
            q.put(("error", f"Fehler beim Laden des Modells: {traceback.format_exc()}"))

    while True:
        event, values = window.read(timeout=200)
        
        # Poll queue
        if update_q:
            try:
                msg_type, *payload = update_q.get_nowait()
                if msg_type == 'log': window['-LOG-'].update(payload[0] + "\n", append=True)
                elif msg_type == 'preprocess_progress': window['-STATUS-'].update(f'Preprocessing... Datei {payload[0]} von {payload[1]}'); window['-PROG_ALL-'].update_bar(payload[0], payload[1])
                elif msg_type == 'transcription_progress': window['-STATUS-'].update(f'Transkribiere... Segment {payload[0]} von {payload[1]}'); window['-PROG_ALL-'].update_bar(payload[0], payload[1])
                elif msg_type in ('stopped', 'error', 'processing_complete'):
                    if msg_type == 'processing_complete': gui_log(update_q, "DONE", f"Prozess abgeschlossen. Zip: {payload[3]}")
                    elif msg_type == 'error': gui_log(update_q, "ERROR", payload[0])
                    window['-START-'].update(disabled=False); window['-PAUSE-'].update(visible=False); window['-STOP-'].update(visible=False); update_q = None; worker_thread = None
            except queue.Empty: pass
        
        if event == sg.WIN_CLOSED: break
        
        if event == '-ADD-':
            paths = sg.popup_get_file("Wähle Dateien", multiple_files=True);
            if paths: add_files_to_list(paths.split(";"))
        
        elif event == '+DRAG_DROP+':
             add_files_to_list(values['+DRAG_DROP+'])
        
        elif event == '-REMOVE-':
            for s in values['-FILES-']: file_set.discard(s)
            window['-FILES-'].update(sorted(list(file_set))); update_start_button()
            
        elif event == '-CLEAR-':
            file_set.clear(); window['-FILES-'].update([]); update_start_button()

        elif event == '-LOAD-':
            model_loaded = False; update_start_button(); update_q = queue.Queue()
            gui_log(update_q, "INFO", "Lade Ressourcen im Hintergrund...")
            threading.Thread(target=load_transcriber_worker, args=(update_q, values["-ENGINE-"], values["-MODEL-"]), daemon=True).start()

        elif event == '-START-':
            update_q = queue.Queue(); stop_event, pause_event = threading.Event(), threading.Event()
            window['-START-'].update(disabled=True); window['-PAUSE-'].update(visible=True); window['-STOP-'].update(visible=True)
            worker_thread = threading.Thread(target=pipeline_worker, args=(sorted(list(file_set)), values["-ENGINE-"], values["-MODEL-"], values["-THREADS-"], 'männlich' if values['-G_M-'] else 'weiblich' if values['-G_F-'] else 'alle', values['-INC_UNKNOWN-'], update_q, stop_event, pause_event, classifier), daemon=True)
            worker_thread.start()
        
        elif event == '-PAUSE-':
            if pause_event:
                if pause_event.is_set(): pause_event.clear(); window['-PAUSE-'].update("Pause")
                else: pause_event.set(); window['-PAUSE-'].update("Fortsetzen")

        elif event == '-STOP-':
            if stop_event: stop_event.set()

        elif event == '-CALIB-':
            calibration_dialog(classifier)
        
        elif event == '-SAVELOG-':
            path = sg.popup_get_file("Log speichern", save_as=True, default_path=f"log_{TIMESTAMP}.txt", file_types=(("Log Datei", "*.log"),))
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f: f.write(window['-LOG-'].get())
                    sg.popup("Log gespeichert!")
                except Exception as e: sg.popup_error(f"Fehler: {e}")

        # Update model_loaded flag from queue
        if update_q and not model_loaded:
             try:
                msg_type, _ = update_q.get_nowait()
                if msg_type == 'model_loaded': model_loaded = True; update_start_button()
             except queue.Empty: pass
             
    window.close()
    
if __name__ == "__main__":
    main()