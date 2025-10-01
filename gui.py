import os
import queue
import sys
import threading
import datetime
import traceback
import io

import PySimpleGUI as sg
import numpy as np
from pydub.playback import play
import matplotlib.pyplot as plt

# Lokale Modulimporte
from config import (
    DEFAULT_PREPROCESS_THREADS, WHISPER_MODEL_OPTIONS, BASE_OUTPUT_DIR, MIN_SEGMENT_SEC, MAX_SEGMENT_SEC, MIN_DBFS, MIN_SNR_DB
)
from utils import gui_log, set_log_file
from gender_classifier import GenderClassifier
from transcription import Transcriber
from pipeline import pipeline_worker, export_session

# Matplotlib für dunkles Theme konfigurieren
plt.style.use('dark_background')

def generate_waveform_image(segment):
    """Erzeugt ein PNG-Bild der Wellenform für ein Audio-Segment."""
    samples = np.array(segment.get_array_of_samples())
    
    # Normalisiere die Samples für eine bessere Visualisierung
    if samples.dtype == np.int16:
        max_val = np.iinfo(np.int16).max
    elif samples.dtype == np.int32:
        max_val = np.iinfo(np.int32).max
    else:
        max_val = np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1.0
        
    normalized_samples = samples.astype(np.float32) / max_val

    fig = plt.figure(figsize=(8, 2), dpi=100)
    ax = fig.add_subplot(111)
    
    ax.plot(normalized_samples, color='#61b0ff')
    
    ax.plot(samples, color='#61b0ff')
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def show_error_and_exit(message: str, exit_now: bool = True):
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

    advanced_options = [
        [sg.Text("Min. Segmentlänge (s):", s=20), sg.Spin([i/10 for i in range(10, 201, 5)], initial_value=MIN_SEGMENT_SEC, resolution=0.5, size=(5,1), key='-MIN_SEC-')],
        [sg.Text("Max. Segmentlänge (s):", s=20), sg.Spin([i for i in range(5, 61)], initial_value=MAX_SEGMENT_SEC, size=(5,1), key='-MAX_SEC-')],
        [sg.Text("Stille-Schwelle (dBFS):", s=20), sg.Spin([i for i in range(-80, -20)], initial_value=MIN_DBFS, size=(5,1), key='-MIN_DBFS-')],
        [sg.Text("SNR-Schwelle (dB):", s=20), sg.Spin([i for i in range(0, 31)], initial_value=MIN_SNR_DB, size=(5,1), key='-MIN_SNR-')]
    ]

    layout = [
        [sg.Text("TTS Datensatz Formatter v12.2", font=("Helvetica", 16))],
        [sg.Column(left_col), sg.VSeperator(), sg.Column(right_col)],
        [sg.Checkbox('Erweiterte Einstellungen anzeigen', key='-SHOW_ADV-', enable_events=True)],
        [sg.Column(advanced_options, key='-ADV_COL-', visible=False, background_color='#404040')],
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
    window = sg.Window("TTS Toolkit v12.2 (Optimierte Pipeline)", layout, finalize=True)
    window['-FILES-'].bind('<Drop>', '+DRAG_DROP')
    window['-DROP_TEXT-'].bind('<Drop>', '+DRAG_DROP')
    return window

def open_editor_window(meta_data, session_timestamp, q):
    """Öffnet das Editor-Fenster zur Nachbearbeitung der Segmente."""
    headings = ["Originaldatei", "Dauer (s)", "Transkript"]
    table_data = [[os.path.basename(item['orig_file']), f"{item['duration_sec']:.2f}", item['transcript']] for item in meta_data]

    editor_layout = [
        [sg.Text("Segment-Editor", font=("Helvetica", 16))],
        [sg.Table(values=table_data, headings=headings, key='-TABLE-',
                  auto_size_columns=False, col_widths=[20, 10, 60],
                  justification='left', enable_events=True, num_rows=15, select_mode=sg.TABLE_SELECT_MODE_BROWSE)],
        [sg.Image(key='-WAVEFORM-', background_color='black', size=(800, 200))],
        [sg.Button("Abspielen", key='-PLAY-'), sg.Button("Löschen", key='-DELETE-')],
        [sg.Text("Transkript bearbeiten:")],
        [sg.Multiline("", key='-TRANSCRIPT-', size=(95, 5), enable_events=True)],
        [sg.Button("Änderungen speichern", key='-SAVE-', disabled=True)],
        [sg.Frame("Export", [[sg.Button("Exportieren & Beenden", key='-EXPORT-', button_color=('white', 'green'))]])]
    ]

    window = sg.Window("Segment-Editor", editor_layout, finalize=True)
    selected_row_index = None
    dirty = False

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            if sg.popup_yes_no("Wollen Sie wirklich ohne Exportieren schließen? Alle Änderungen gehen verloren.") == 'Yes':
                break

        elif event == '-TABLE-':
            if values['-TABLE-']:
                if dirty and sg.popup_yes_no("Es gibt nicht gespeicherte Änderungen. Trotzdem fortfahren?") == 'No':
                    window['-TABLE-'].update(select_rows=[selected_row_index] if selected_row_index is not None else [])
                    continue
                selected_row_index = values['-TABLE-'][0]
                transcript = meta_data[selected_row_index]['transcript']
                window['-TRANSCRIPT-'].update(transcript)
                
                # Wellenform aktualisieren
                waveform_data = generate_waveform_image(meta_data[selected_row_index]['segment'])
                window['-WAVEFORM-'].update(data=waveform_data)

                window['-SAVE-'].update(disabled=True)
                dirty = False
        
        elif event == '-TRANSCRIPT-':
            if selected_row_index is not None:
                dirty = True
                window['-SAVE-'].update(disabled=False)

        elif event == '-PLAY-':
            if selected_row_index is not None:
                try:
                    segment_to_play = meta_data[selected_row_index]['segment']
                    threading.Thread(target=play, args=(segment_to_play,), daemon=True).start()
                except Exception as e:
                    sg.popup_error(f"Fehler beim Abspielen: {e}")

        elif event == '-SAVE-':
            if selected_row_index is not None and dirty:
                new_transcript = values['-TRANSCRIPT-']
                meta_data[selected_row_index]['transcript'] = new_transcript
                table_data[selected_row_index][2] = new_transcript
                window['-TABLE-'].update(values=table_data)
                window['-SAVE-'].update(disabled=True)
                dirty = False
        
        elif event == '-DELETE-':
            if selected_row_index is not None and sg.popup_yes_no("Wollen Sie dieses Segment wirklich löschen?") == 'Yes':
                meta_data.pop(selected_row_index)
                table_data.pop(selected_row_index)
                window['-TABLE-'].update(values=table_data)
                selected_row_index = None
                window['-TRANSCRIPT-'].update("")
                window['-WAVEFORM-'].update(data=None)
                dirty = False

        elif event == '-EXPORT-':
            if dirty and sg.popup_yes_no("Es gibt nicht gespeicherte Änderungen. Trotzdem exportieren?") == 'No':
                continue
            zip_path = export_session(session_timestamp, meta_data, q)
            if zip_path:
                sg.popup(f"Export erfolgreich abgeschlossen!\nDatei gespeichert unter: {zip_path}")
            else:
                sg.popup_error("Export fehlgeschlagen. Überprüfen Sie die Logs.")
            break

    window.close()

def start_gui():
    """Initialisiert und startet die Haupt-GUI-Schleife der Anwendung."""
    
    window = create_main_window()
    classifier = GenderClassifier()
    file_set = set()
    worker_thread, update_q, stop_event, pause_event = None, None, None, None
    model_loaded = False
    
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
            Transcriber(engine, model_name)
            model_loaded = True
            q.put(("model_loaded", True))
        except Exception:
            model_loaded = False
            q.put(("error", traceback.format_exc()))

    main_window_active = True
    while main_window_active:
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
                    
                    elif msg_type == 'processing_complete':
                        gui_log(update_q, "DONE", "Pipeline-Verarbeitung abgeschlossen. Öffne Editor...")
                        window.hide()
                        open_editor_window(payload[0], session_timestamp, update_q)
                        main_window_active = False
                        break

                    elif msg_type in ('stopped', 'error'):
                        is_error = msg_type == 'error'
                        if is_error:
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
        
        if not main_window_active:
            break

        if event == sg.WIN_CLOSED:
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

        elif event == '-SHOW_ADV-':
            window['-ADV_COL-'].update(visible=values['-SHOW_ADV-'])

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
            
            try:
                advanced_settings = {
                    "min_sec": float(values['-MIN_SEC-']),
                    "max_sec": float(values['-MAX_SEC-']),
                    "min_dbfs": int(values['-MIN_DBFS-']),
                    "min_snr_db": float(values['-MIN_SNR-'])
                }
            except (ValueError, TypeError):
                sg.popup_error("Bitte geben Sie gültige Zahlen für die erweiterten Einstellungen ein.")
                window['-START-'].update(disabled=False)
                continue

            worker_thread = threading.Thread(
                target=pipeline_worker,
                args=(
                    sorted(list(file_set)), values["-ENGINE-"], values["-MODEL-"],
                    int(values["-THREADS-"]), gender_choice, update_q,
                    stop_event, pause_event, classifier, session_timestamp,
                    advanced_settings
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
    
    if worker_thread and worker_thread.is_alive():
        if stop_event:
            stop_event.set()
        worker_thread.join(timeout=2)
    window.close()se()