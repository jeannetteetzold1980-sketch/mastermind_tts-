import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from audio_processing import (
    preprocess_segment, safe_load_audio, segment_audio, split_into_segments, normalize_text
)
from config import BASE_OUTPUT_DIR
from gender_classifier import GenderClassifier
from transcription import Transcriber, TranscriptCache
from utils import gui_log


def export_session(session_id: str, meta_data: list, q: queue.Queue = None):
    """Exports the final metadata and audio files into a ZIP archive."""
    try:
        gui_log(q, "INFO", "📦 Starte finalen Export...")
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
        if q:
            q.put(("export_finished", zip_path))
        return zip_path
    except Exception:
        tb = traceback.format_exc()
        gui_log(q, "ERROR", f"Export fehlgeschlagen:\n{tb}")
        if q:
            q.put(("export_failed", tb))
        return None

def pipeline_worker(
    file_list, model_engine, model_name, threads, gender_filter, q, 
    stop_event, pause_event, classifier: GenderClassifier, session_timestamp: str,
    advanced_settings: dict
):
    """
    Die Haupt-Worker-Funktion, die die gesamte Verarbeitungspipeline für eine Liste von Dateien ausführt.
    """
    tmp_session, out_session = None, None
    try:
        gui_log(q, "INFO", "Pipeline startet...")
        session_id = session_timestamp
        tmp_session = os.path.join(BASE_OUTPUT_DIR, f"tmp_{session_id}")
        os.makedirs(tmp_session, exist_ok=True)

        # SCHRITT 1: GENDER-FILTERUNG
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

        # SCHRITT 2: SEGMENTIERUNG
        gui_log(q, "INFO", "📊 SCHRITT 2: Intelligente Audio-Segmentierung...")
        all_segments = []
        for i, filepath in enumerate(filtered_files):
            if stop_event.is_set(): raise InterruptedError()
            try:
                audio, sr = segment_audio(filepath)
                segments = split_into_segments(
                    audio, filepath, tmp_session,
                    min_sec=advanced_settings['min_sec'],
                    max_sec=advanced_settings['max_sec'],
                    min_dbfs=advanced_settings['min_dbfs']
                )
                all_segments.extend(segments)
                if segments:
                    gui_log(q, "INFO", f"✓ {os.path.basename(filepath)}: {len(segments)} Segmente gefunden.")
                else:
                    gui_log(q, "WARN", f"Keine Segmente für {os.path.basename(filepath)} gefunden.")
            except Exception as e:
                gui_log(q, "WARN", f"Segmentierung fehlgeschlagen für {os.path.basename(filepath)}: {e}")
            q.put(("progress_update", "2. Segmentierung", i + 1, len(filtered_files)))
        
        if not all_segments:
            raise ValueError("Keine verwendbaren Segmente nach Segmentierung gefunden!")
        gui_log(q, "INFO", f"✓ Segmentierung abgeschlossen: {len(all_segments)} Segmente insgesamt.")

        # SCHRITT 3: PREPROCESSING
        gui_log(q, "INFO", "⚙️ SCHRITT 3: Audio-Preprocessing...")
        processed_segments = []
        rejected_count = 0
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_seg = {executor.submit(preprocess_segment, seg): seg for seg in all_segments}
            for i, future in enumerate(as_completed(future_to_seg)):
                if stop_event.is_set(): raise InterruptedError()
                try:
                    result = future.result()
                    if result.get('quality_rejected', False):
                        rejected_count += 1
                        gui_log(q, "WARN", f"Segment verworfen: {result.get('reject_reason', 'unbekannt')}")
                    else:
                        processed_segments.append(result)
                except Exception as e:
                    gui_log(q, "WARN", f"Preprocessing fehlgeschlagen für Segment: {e}")
                    rejected_count += 1
                q.put(("progress_update", "3. Preprocessing", i + 1, len(all_segments)))
        
        gui_log(q, "INFO", f"✓ Preprocessing abgeschlossen: {len(processed_segments)} akzeptiert, {rejected_count} verworfen.")
        if not processed_segments:
            raise ValueError("Keine Segmente haben die Qualitäts-Checks bestanden!")

        # SCHRITT 4: TRANSKRIPTION
        gui_log(q, "INFO", "✍️ SCHRITT 4: Transkription mit Whisper...")
        transcriber = Transcriber(engine_preference=model_engine, model_name=model_name)
        gui_log(q, "INFO", f"Transcriber-Engine: {transcriber.engine}")
        cache = TranscriptCache()
        
        meta_data = []
        for i, seg in enumerate(processed_segments):
            if stop_event.is_set(): break
            while pause_event.is_set(): time.sleep(0.1)
            
            cached = cache.get(seg['sha'])
            if cached:
                seg['transcript'] = cached
                gui_log(q, "DEBUG", f"Cache-Hit für Segment {i+1}")
            else:
                seg_path = seg['wav_path']
                seg['segment'].export(seg_path, format="wav")
                res = transcriber.transcribe(seg_path)
                seg['transcript'] = normalize_text(res.get('text', ''))
                
                if seg['transcript']:
                    cache.set(seg['sha'], seg['transcript'])
                
                try: os.remove(seg_path)
                except OSError: pass
            
            if seg.get('transcript'):
                meta_data.append(seg)
            
            q.put(("progress_update", "4. Transkription", i + 1, len(processed_segments)))
        
        if stop_event.is_set(): raise InterruptedError()
        if not meta_data: raise ValueError("Keine nutzbaren Segmente nach Transkription übrig.")
        gui_log(q, "INFO", f"✓ Transkription abgeschlossen: {len(meta_data)} transkribierte Segmente.")

        # SCHRITT 5: ÜBERGABE AN EDITOR
        gui_log(q, "INFO", "Verarbeitung abgeschlossen. Übergebe Daten an den Editor.")
        q.put(("processing_complete", meta_data))

    except InterruptedError:
        gui_log(q, "INFO", "Prozess vom Benutzer gestoppt.")
        q.put(('stopped',))
    except Exception:
        gui_log(q, "ERROR", f"Ein unerwarteter Fehler ist aufgetreten:\n{traceback.format_exc()}")
        q.put(('error', traceback.format_exc()))
    finally:
        if tmp_session and os.path.exists(tmp_session):
            shutil.rmtree(tmp_session, ignore_errors=True)
