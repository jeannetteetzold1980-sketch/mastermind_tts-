# tests/test_pipeline.py

import queue
import threading
import os
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import pipeline_worker

def test_pipeline_worker_full_run(tmp_path, temp_wav_file, mock_transcriber, classifier_instance):
    """
    Testet den kompletten `pipeline_worker` von Anfang bis Ende.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(temp_wav_file, input_dir)
    
    file_list = [str(input_dir / os.path.basename(temp_wav_file))]
    
    q = queue.Queue()
    stop_event = threading.Event()
    pause_event = threading.Event()
    
    # Rufe den Worker auf. Er läuft synchron in diesem Test.
    pipeline_worker(
        file_list=file_list,
        model_engine='mocked',
        model_name='mocked',
        preprocess_threads=1,
        gender_filter='alle',
        update_q=q,
        stop_event=stop_event,
        pause_event=pause_event,
        classifier=classifier_instance,
        # Wir übergeben den mock_transcriber, um den echten zu umgehen
        _test_transcriber_obj=mock_transcriber
    )

    results = []
    while not q.empty():
        results.append(q.get_nowait())

    # Hat der Worker eine "processing_complete" Nachricht gesendet?
    complete_messages = [msg for msg in results if msg[0] == 'processing_complete']
    assert len(complete_messages) == 1, "Der Worker sollte genau eine 'complete' Nachricht senden."
    
    zip_path = complete_messages[0][4]
    assert os.path.exists(zip_path), "Die resultierende ZIP-Datei wurde nicht erstellt."

    extract_dir = tmp_path / "extracted"
    shutil.unpack_archive(zip_path, extract_dir)
    
    metadata_path = extract_dir / "metadata.tsv"
    assert metadata_path.exists(), "metadata.tsv fehlt im Ergebnis."
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "test_1_" in content
        assert "dies ist ein test" in content

def test_pipeline_worker_stop_event(temp_wav_file, mock_transcriber, classifier_instance):
    """Testet, ob der Worker auf ein stop_event sauber reagiert und keine Ergebnisse schreibt."""
    q = queue.Queue()
    stop_event = threading.Event()
    stop_event.set() # Setze das Stopp-Signal VOR dem Start
    
    pipeline_worker([], 'mock', 'mock', 1, 'alle', q, stop_event, threading.Event(), classifier_instance, mock_transcriber)

    results = []
    while not q.empty():
        results.append(q.get_nowait())
    
    stopped_messages = [msg for msg in results if msg[0] == 'stopped']
    assert len(stopped_messages) >= 1, "Der Worker sollte eine 'stopped' Nachricht senden."

    complete_messages = [msg for msg in results if msg[0] == 'processing_complete']
    assert len(complete_messages) == 0, "Der Worker sollte im gestoppten Zustand keine 'complete' Nachricht senden."