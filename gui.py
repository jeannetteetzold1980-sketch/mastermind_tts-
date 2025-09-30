import os
import queue
import sys
import threading
import datetime
import traceback

import PySimpleGUI as sg
import numpy as np

# Lokale Modulimporte
from config import (
    DEFAULT_PREPROCESS_THREADS,
    WHISPER_MODEL_OPTIONS,
    BASE_OUTPUT_DIR
)
from utils import gui_log, set_log_file
from gender_classifier import GenderClassifier, HAVE_SKLEARN
from transcription import Transcriber
from pipeline import pipeline_worker


def show_error_and_exit(message: str, exit_now: bool = True):
    """Zeigt eine Fehlermeldung an und beendet das Programm kontrolliert."""
    sg.popup_error(message, title="Schwerwiegender Fehler")
    if exit_now:
        sys.exit(1)

def create_main_window():
    """Erstellt das Hauptfenster der Anwendung."""
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
        ], element_justification='center', key='-ACTION_FRAME-', visible=False)],
        [sg.Frame("Logs", [
            [sg.Multiline("", size=(120, 16), key="-LOG-", autoscroll=True, disabled=True)]
        ])]
    ]
    window = sg.Window("TTS Toolkit v12.0 (Optimierte Pipeline)", layout, finalize=True)
    window['-FILES-'].bind('<Drop>', '+DRAG_DROP')
    window['-DROP_TEXT-'].bind('<Drop>', '+DRAG_DROP')
    return window


def calibration_dialog(classifier: GenderClassifier):
    """Zeigt den Dialog zur Kalibrierung des GenderClassifiers an."""
    if not HAVE_SKLEARN:
        sg.popup_error("scikit-learn ist nicht installiert. Kalibrierung nicht möglich.")
        return

    layout = [
        [sg.Text("Gender-Kalibrierung")],
        [sg.FilesBrowse("Dateien auswählen", key="-CAL_FILES-"), sg.Button("Laden", key="-CAL_LOAD-")],
        [sg.Table(values=[], headings=["Datei", "Label"], key="-CAL_TABLE-", auto_size_columns=False, col_widths=[80, 12], num_rows=10, select_mode=sg.TABLE_SELECT_MODE_EXTENDED, enable_events=True)],
        [sg.Button("Als Männlich", key="-CAL_M-"), sg.Button("Als Weiblich", key="-CAL_F-")],
        [sg.Button("Training starten\n(min. 2 pro Klasse)", key="-CAL_TRAIN-"), sg.Button("Schließen", key="-CAL_CLOSE-")],
        [sg.Text("", key="-CAL_STATUS-")]
    ]
    win = sg.Window("Gender-Kalibrierung", layout, modal=True, finalize=True)
    files, labels = [], []
    
    def refresh_table():
        rows = [[os.path.basename(files[i]), labels[i]] for i in range(len(files))]
        win["-CAL_TABLE-"].update(values=rows)
        m = labels.count("männlich")
        f = labels.count("weiblich")
        u = labels.count("unlabeled")
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
            pairs = [(files[i], labels[i]) for i in range(len(files)) if labels[i] != "unlabeled"]
            if sum(1 for _,l in pairs if l=="männlich") < 2 or sum(1 for _,l in pairs if l=="weiblich") < 2:
                sg.popup_error("Bitte min. 2 Beispiele pro Geschlecht markieren.")
                continue
            
            try:
                win["-CAL_TRAIN-"].update(disabled=True)
                res = classifier.train(pairs)
                cv_scores, report = res.get("cv_scores", []), res.get("report", {})
                txt = f"Training abgeschlossen.\nCV Accuracy: {np.mean(cv_scores):.3f}\n\nReport:\n"
                for cls, metrics in report.items():
                    if isinstance(metrics, dict):
                        txt += f"{cls}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}\n"
                sg.popup_scrolled(txt, title="Trainingsergebnis")
                break 
            except Exception as e:
                sg.popup_error(f"Fehler beim Training: {e}")
            finally:
                win["-CAL_TRAIN-"].update(disabled=False)
    
    win.close()


def start_gui():
    """Initialisiert und startet die Haupt-GUI-Schleife der Anwendung."""
    
    # Globale Zustandsvariablen der GUI
    window = create_main_window()
    classifier = GenderClassifier()
    file_set = set()
    worker_thread, update_q, stop_event, pause_event = None, None, None, None
    model_loaded = False
    
    # Setze Log-Datei für diese Sitzung
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(BASE_OUTPUT_DIR, f"run_{session_timestamp}.log")
    set_log_file(log_file_path)

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
        nonlocal model_loaded
        try:
            gui_log(q, "INFO", f"Lade Transcriber-Modell '{model_name}' mit Engine '{engine}'...")
            Transcriber(engine, model_name) # Nur zum Testen des Ladens
            model_loaded = True
            q.put(("model_loaded", True))
        except Exception:
            model_loaded = False
            q.put(("error", traceback.format_exc()))

    # Haupt-Event-Schleife
    while True:
        event, values = window.read(timeout=100)
        
        # GUI-Updates aus dem Worker-Thread verarbeiten
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
                        is_error = msg_type == 'error'
                        if msg_type == 'processing_complete':
                            gui_log(update_q, "DONE", f"Prozess abgeschlossen. Zip: {payload[3]}")
                        elif is_error:
                            gui_log(update_q, "ERROR", payload[0])
                        
                        window['-STATUS-'].update("Fehler!" if is_error else "Bereit.")
                        window['-PROG_ALL-'].update(0, 1)
                        window['-START-'].update(disabled=False)
                        window['-ACTION_FRAME-'].update(visible=False)
                        worker_thread = None
                    
                    elif msg_type == 'model_loaded':
                        model_loaded = True
                        update_start_button()
                        gui_log(update_q, "INFO", "Ressourcen bereit. Pipeline kann gestartet werden.")
            
            except queue.Empty:
                pass
        
        if event == sg.WIN_CLOSED:
            if worker_thread and worker_thread.is_alive():
                if stop_event: stop_event.set()
                worker_thread.join(timeout=2)
            break

        if event == '-ADD-':
            paths_str = sg.popup_get_file("Wähle Dateien", multiple_files=True, file_types=(("Audio Files", "*.*"),))
            if paths_str: add_files_to_list(paths_str.split(";"))
        
        elif event.endswith('+DRAG_DROP'):
            add_files_to_list(values[event.split('+')[0]])

        elif event == '-REMOVE-':
            for s in values['-FILES-']: file_set.discard(s)
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
            
            gender_choice = 'männlich' if values['-G_M-'] else 'weiblich' if values['-G_F-'] else 'alle'
            
            worker_thread = threading.Thread(
                target=pipeline_worker,
                args=(
                    sorted(list(file_set)), values["-ENGINE-"], values["-MODEL-"],
                    int(values["-THREADS-"]), gender_choice, update_q,
                    stop_event, pause_event, classifier, session_timestamp
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
            calibration_dialog(classifier)
            
        elif event == '-SAVELOG-':
            path = sg.popup_get_file("Log speichern", save_as=True, default_path=f"log_{session_timestamp}.txt", file_types=(("Log Datei", "*.log"),))
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(window['-LOG-'].get())
                    sg.popup("Log gespeichert!")
                except Exception as e:
                    sg.popup_error(f"Fehler beim Speichern des Logs: {e}")
    
    window.close()
