
import argparse
import os
import queue
import threading
import time
from datetime import datetime

from pipeline import pipeline_worker, export_session
from gender_classifier import GenderClassifier
from utils import ConsoleLogHandler

def main():
    parser = argparse.ArgumentParser(description="MasterMind-TTS CLI")
    parser.add_argument("input_files", nargs="+", help="List of audio files to process.")
    parser.add_argument("--model_engine", default="whisper_jax", help="Transcription model engine.")
    parser.add_argument("--model_name", default="large-v3", help="Transcription model name.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for preprocessing.")
    parser.add_argument("--gender_filter", default="alle", choices=["alle", "male", "female"], help="Filter by gender.")
    parser.add_argument("--min_sec", type=float, default=3.0, help="Minimum segment length in seconds.")
    parser.add_argument("--max_sec", type=float, default=25.0, help="Maximum segment length in seconds.")
    parser.add_argument("--min_dbfs", type=int, default=-40, help="Minimum dBFS for a segment.")

    args = parser.parse_args()

    q = queue.Queue()
    log_thread = threading.Thread(target=ConsoleLogHandler, args=(q,), daemon=True)
    log_thread.start()

    stop_event = threading.Event()
    pause_event = threading.Event()
    
    classifier = GenderClassifier()
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    advanced_settings = {
        "min_sec": args.min_sec,
        "max_sec": args.max_sec,
        "min_dbfs": args.min_dbfs,
    }

    # Da die Verarbeitung in einem eigenen Thread stattfindet, müssen wir auf das Ergebnis warten.
    result_queue = queue.Queue()

    def worker_wrapper():
        pipeline_worker(
            file_list=args.input_files,
            model_engine=args.model_engine,
            model_name=args.model_name,
            threads=args.threads,
            gender_filter=args.gender_filter,
            q=q,
            stop_event=stop_event,
            pause_event=pause_event,
            classifier=classifier,
            session_timestamp=session_timestamp,
            advanced_settings=advanced_settings,
        )
        # Wir verwenden die Haupt-Queue, um das Ergebnis zu signalisieren
        # In einer echten CLI-Anwendung könnte man das anders lösen.
        while True:
            try:
                message = q.get_nowait()
                if message[0] == 'processing_complete':
                    result_queue.put(message[1])
                    break
                elif message[0] in ('error', 'stopped'):
                    result_queue.put(None)
                    break
            except queue.Empty:
                time.sleep(0.1)


    worker_thread = threading.Thread(target=worker_wrapper)
    worker_thread.start()
    worker_thread.join()

    # Ergebnisse abrufen und exportieren
    final_metadata = result_queue.get()
    if final_metadata:
        export_session(session_timestamp, final_metadata, q)

if __name__ == "__main__":
    main()
