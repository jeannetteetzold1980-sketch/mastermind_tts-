
import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import eventlet
import threading
import queue

# Import your existing pipeline components
from pipeline import pipeline_worker, export_session
from gender_classifier import GenderClassifier
from utils import ConsoleLogHandler # Assuming you want logs in the web UI
from config import *

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Background Pipeline Thread ---
thread = None
thread_stop_event = threading.Event()
task_queue = queue.Queue()

def pipeline_thread_wrapper(files_to_process, advanced_settings):
    """Wraps the pipeline worker to emit progress over SocketIO."""
    
    log_q = queue.Queue()
    
    # Redirect logs to the web UI
    def web_log(q, level, message):
        socketio.emit('log', {'level': level, 'message': message})
    
    # Replace the console logger with our web logger
    # This is a bit of a hack, ideally the pipeline would be more pluggable
    import utils
    original_gui_log = utils.gui_log
    utils.gui_log = web_log

    try:
        session_timestamp = "web_session" # Or generate a real one
        classifier = GenderClassifier()
        
        # The pipeline_worker expects a queue for updates, let's create one
        update_q = queue.Queue()

        # We need a separate thread to monitor the update queue
        def progress_monitor():
            while True:
                try:
                    msg_type, *payload = update_q.get(timeout=1.0)
                    if msg_type == 'progress_update':
                        stage, current, total = payload
                        socketio.emit('progress', {'stage': stage, 'current': current, 'total': total})
                    elif msg_type == 'log':
                         socketio.emit('log', {'level': 'INFO', 'message': payload[0]})
                    elif msg_type == 'processing_complete':
                        socketio.emit('log', {'level': 'DONE', 'message': "Verarbeitung abgeschlossen!"})
                        # In a real app, you'd handle the results (payload[0]) here
                        break 
                    elif msg_type in ('stopped', 'error'):
                        socketio.emit('log', {'level': 'ERROR', 'message': "Ein Fehler ist aufgetreten."})
                        break
                except queue.Empty:
                    if thread_stop_event.is_set():
                        break
        
        monitor_thread = threading.Thread(target=progress_monitor)
        monitor_thread.start()

        pipeline_worker(
            file_list=files_to_process,
            model_engine=WHISPER_ENGINE,
            model_name=WHISPER_MODEL,
            threads=DEFAULT_PREPROCESS_THREADS,
            gender_filter='alle', # Hardcoded for now
            q=update_q,
            stop_event=thread_stop_event,
            pause_event=threading.Event(), # Not implemented in web UI
            classifier=classifier,
            session_timestamp=session_timestamp,
            advanced_settings=advanced_settings
        )
    except Exception as e:
        socketio.emit('log', {'level': 'ERROR', 'message': f"Schwerwiegender Fehler in der Pipeline: {e}"})
    finally:
        # Restore original logger
        utils.gui_log = original_gui_log
        socketio.emit('pipeline_finished')


# --- Flask Routes ---
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    
    upload_dir = os.path.join(app.root_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    processed_files = []
    for file in files:
        if file.filename == '':
            continue
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)
        processed_files.append(filepath)
        
    return jsonify({'message': f'{len(processed_files)} Dateien hochgeladen.', 'files': processed_files})

# --- Socket.IO Events ---
@socketio.on('start_processing')
def start_processing(data):
    """Start the background processing thread."""
    global thread
    if thread and thread.is_alive():
        emit('log', {'level': 'WARN', 'message': 'Die Verarbeitung läuft bereits.'})
        return

    files = data.get('files')
    if not files:
        emit('log', {'level': 'ERROR', 'message': 'Keine Dateien zum Verarbeiten ausgewählt.'})
        return

    # Get advanced settings from the form
    advanced_settings = {
        "min_sec": float(data.get('min_sec', MIN_SEGMENT_SEC)),
        "max_sec": float(data.get('max_sec', MAX_SEGMENT_SEC)),
        "min_dbfs": int(data.get('min_dbfs', MIN_DBFS)),
    }

    emit('log', {'level': 'INFO', 'message': 'Starte Verarbeitung im Hintergrund...'})
    
    thread_stop_event.clear()
    thread = threading.Thread(target=pipeline_thread_wrapper, args=(files, advanced_settings))
    thread.start()

@socketio.on('stop_processing')
def stop_processing():
    """Stop the background thread."""
    global thread
    if thread and thread.is_alive():
        thread_stop_event.set()
        emit('log', {'level': 'WARN', 'message': 'Stopp-Signal gesendet. Warte auf Abschluss des aktuellen Schritts...'})
    else:
        emit('log', {'level': 'INFO', 'message': 'Kein laufender Prozess zum Stoppen.'})


if __name__ == '__main__':
    print("Web GUI gestartet. Öffnen Sie http://127.0.0.1:5000 in Ihrem Browser.")
    socketio.run(app, host='0.0.0.0', port=5000)
