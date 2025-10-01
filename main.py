# main.py — TTS Toolkit Pipeline
import os
import sys
import time
import traceback
import queue
import threading

# Eigene Module
from gender_classifier import GenderClassifier
from audio_processing import segment_audio, split_into_segments, preprocess_segment
from transcription import Transcriber, TranscriptCache
from config import TEMP_SEGMENT_DIR, WHISPER_MODEL, WHISPER_ENGINE, DEVICE, COMPUTE_TYPE


# --- Worker-Funktion ---
def pipeline_worker(task_queue, config=None):
    """
    Pipeline: Gender → Segmentierung → Preprocessing → Transkription → Export
    """
    import queue as pyqueue
    gender_clf = GenderClassifier()
    transcriber = Transcriber(
        engine_preference=WHISPER_ENGINE,
        model_name=WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    cache = TranscriptCache()

    while True:
        try:
            filepath = task_queue.get(timeout=1)
        except pyqueue.Empty:
            break

        try:
            print(f"\n[Worker] Starte Verarbeitung: {filepath}")

            # 1. Gender-Erkennung
            y, sr = None, None
            try:
                import soundfile as sf
                y, sr = sf.read(filepath, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
            except Exception as e:
                print(f"[Gender] Laden fehlgeschlagen: {e}")

            if y is not None:
                gender, conf = gender_clf.predict(y, sr)
                print(f"[Pipeline] Gender erkannt: {gender} ({conf:.2f})")
            else:
                gender, conf = "unbekannt", 0.0

            # 2. Segmentierung
            audio, sr = segment_audio(filepath)
            segments = split_into_segments(audio, filepath, TEMP_SEGMENT_DIR)
            print(f"[Pipeline] {len(segments)} Segmente erstellt.")

            # 3. Preprocessing
            processed_segments = []
            for seg in segments:
                seg = preprocess_segment(seg)
                if not seg.get("quality_rejected", False):
                    processed_segments.append(seg)
                else:
                    print(f"[Preprocessing] Segment verworfen: {seg.get('reject_reason')}")
            print(f"[Pipeline] {len(processed_segments)} Segmente nach Preprocessing.")

            # 4. Transkription
            texts = []
            for seg in processed_segments:
                sha = seg["sha"]
                cached = cache.get(sha)
                if cached:
                    texts.append(cached)
                    continue

                seg["segment"].export(seg["wav_path"], format="wav")
                res = transcriber.transcribe(seg["wav_path"])
                txt = res.get("text", "").strip()
                if txt:
                    cache.set(sha, txt)
                    texts.append(txt)
            print(f"[Pipeline] Transkription abgeschlossen ({len(texts)} Texte).")

            # 5. Export
            out_path = os.path.splitext(filepath)[0] + "_transcript.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            print(f"[Pipeline] Export fertig: {out_path}")

        except Exception as e:
            print(f"[Worker] Fehler bei {filepath}: {e}")
            traceback.print_exc()
        finally:
            task_queue.task_done()


# --- Main ---
def main():
    q = queue.Queue()

    # Eingabedateien (Beispiel: alle WAV im Ordner "input")
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    if not files:
        print("Keine Eingabedateien gefunden im Ordner 'input'.")
        return

    for f in files:
        q.put(f)

    worker = threading.Thread(target=pipeline_worker, args=(q, None), daemon=True)
    worker.start()

    q.join()
    print("\nAlle Dateien verarbeitet.")


if __name__ == "__main__":
    main()
