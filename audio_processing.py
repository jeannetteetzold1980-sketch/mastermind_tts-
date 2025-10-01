import os
import hashlib
import re
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from num2words import num2words
from pydub import AudioSegment, silence
from scipy.signal import butter, filtfilt
from scipy.stats import linregress




def safe_load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Lädt eine Audiodatei sicher und konvertiert sie in Mono-Float32 mit der Ziel-Samplerate."""
    try:
        data, file_sr = sf.read(path, always_2d=False, dtype='float32')
        # In Mono konvertieren, falls Stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # Resamplen, falls notwendig
        if file_sr != sr:
            data = librosa.resample(y=data, orig_sr=file_sr, target_sr=sr)
        return data, sr
    except Exception:
        # Fallback mit librosa, das mehr Formate unterstützt
        y, loaded_sr = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), loaded_sr

def segment_sha1(segment: AudioSegment) -> str:
    """Erzeugt einen SHA1-Hash für ein AudioSegment."""
    return hashlib.sha1(segment.raw_data).hexdigest()

def normalize_text(text: str) -> str:
    """Normalisiert einen Transkriptionstext: Kleinbuchstaben, Zahlen zu Wörtern, keine Sonderzeichen."""
    if not isinstance(text, str):
        return ""

    def number_to_words(match):
        try:
            return num2words(int(match.group(0)), lang='de')
        except Exception:
            return match.group(0)

    text = re.sub(r'\d+', number_to_words, text.strip())
    text = text.lower()
    text = re.sub(r'[^a-zäöüß\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def segment_audio(filepath: str) -> Tuple[AudioSegment, int]:
    """Lädt eine Audiodatei und gibt sie als vollständiges pydub AudioSegment zurück."""
    try:
        y, sr = safe_load_audio(filepath, sr=16000)
        # Konvertiere float32 numpy array zu pcm16 bytes
        pcm16 = (y * 32767).astype(np.int16).tobytes()
        audio = AudioSegment(data=pcm16, sample_width=2, frame_rate=sr, channels=1)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Audio-Laden fehlgeschlagen für {filepath}: {e}")


def _analyze_prosody_at_split(
    start_ms: int,
    f0: np.ndarray,
    rms: np.ndarray,
    times: np.ndarray,
    window_ms: int = 250
) -> Dict[str, float]:
    """
    Analysiert die Prosodie (Tonhöhe & Energie) in einem Fenster vor einem Split-Punkt.
    """
    analysis = {'pitch_slope': 0.0, 'energy_slope': 0.0}
    
    end_time_sec = start_ms / 1000.0
    window_sec = window_ms / 1000.0
    
    candidate_indices = np.where(times < end_time_sec)[0]
    if len(candidate_indices) == 0:
        return analysis 
        
    end_idx = candidate_indices[-1]
    start_idx = np.where(times < (end_time_sec - window_sec))[0]
    start_idx = start_idx[-1] if len(start_idx) > 0 else 0

    if (end_idx - start_idx) < 4:
        return analysis

    f0_window = f0[start_idx:end_idx]
    rms_window = rms[start_idx:end_idx].flatten()
    time_window = times[start_idx:end_idx]
    
    voiced_indices = np.where(f0_window > 0)[0]
    if len(voiced_indices) > 3:
        f0_voiced = f0_window[voiced_indices]
        time_voiced = time_window[voiced_indices]
        pitch_slope, _, _, _, _ = linregress(time_voiced, f0_voiced)
        analysis['pitch_slope'] = pitch_slope if not np.isnan(pitch_slope) else 0.0

    energy_slope, _, _, _, _ = linregress(time_window, rms_window)
    analysis['energy_slope'] = energy_slope if not np.isnan(energy_slope) else 0.0

    return analysis


def split_into_segments(
    audio: AudioSegment, orig_file: str, temp_dir: str,
    min_sec: float, max_sec: float, min_dbfs: int
) -> List[Dict]:
    """Intelligente Segmentierung basierend auf Stille und Prosodie-Analyse."""
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    
    f0, _, _ = librosa.pyin(samples, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=1024, hop_length=256)
    f0 = np.nan_to_num(f0)
    rms = librosa.feature.rms(y=samples, frame_length=1024, hop_length=256)[0]
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
    
    min_silence_for_split = 250
    silent_chunks = silence.detect_silence(
        audio, 
        min_silence_len=min_silence_for_split,
        silence_thresh=min_dbfs
    )
    
    if not silent_chunks:
        if min_sec <= audio.duration_seconds <= max_sec:
             return [{
                "wav_path": os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(orig_file))[0]}_001.wav"),
                "segment": audio, "sha": segment_sha1(audio), "orig_file": orig_file, "duration_sec": audio.duration_seconds
            }]
        else:
            return []

    split_points = []
    last_end = 0
    for start_ms, end_ms in silent_chunks:
        pause_duration = end_ms - start_ms
        prosody = _analyze_prosody_at_split(start_ms, f0, rms, times)
        pitch_slope = prosody['pitch_slope']
        energy_slope = prosody['energy_slope']
        
        score = (pause_duration / 1000.0) * 1.5
        if pitch_slope < 0:
            score += abs(pitch_slope) / 100.0
        if energy_slope < 0:
            score += abs(energy_slope) / 20.0
        
        split_points.append({
            'start_segment_ms': last_end,
            'end_segment_ms': start_ms,
            'split_score': score,
            'pause_duration': pause_duration
        })
        last_end = end_ms
    
    split_points.append({
        'start_segment_ms': last_end,
        'end_segment_ms': len(audio),
        'split_score': 999,
        'pause_duration': 0
    })

    merged_segments = []
    if not split_points:
        return []
    
    current_merge_start = split_points[0]['start_segment_ms']
    current_merge_end = split_points[0]['end_segment_ms']

    MERGE_THRESHOLD = 1.0 

    for i in range(len(split_points) - 1):
        current_duration = (current_merge_end - current_merge_start) / 1000.0
        next_segment_duration = (split_points[i+1]['end_segment_ms'] - split_points[i+1]['start_segment_ms']) / 1000.0
        potential_duration = current_duration + next_segment_duration + (split_points[i]['pause_duration'] / 1000.0)
        
        split_score = split_points[i]['split_score']
        
        should_merge = False
        if potential_duration > max_sec:
            should_merge = False
        elif current_duration < min_sec:
            should_merge = True
        elif split_score < MERGE_THRESHOLD:
            should_merge = True
        
        if should_merge:
            current_merge_end = split_points[i+1]['end_segment_ms']
        else:
            merged_segments.append(audio[current_merge_start:current_merge_end])
            current_merge_start = split_points[i+1]['start_segment_ms']
            current_merge_end = split_points[i+1]['end_segment_ms']
            
    merged_segments.append(audio[current_merge_start:current_merge_end])

    out_segs = []
    base = os.path.splitext(os.path.basename(orig_file))[0]
    
    for idx, seg in enumerate(merged_segments):
        dur = len(seg) / 1000.0
        dbfs = seg.dBFS if seg.duration_seconds > 0 else -100.0
        
        if min_sec <= dur <= max_sec and dbfs > (min_dbfs - 5):
            sha = segment_sha1(seg)
            fname = f"{base}_{idx+1:03d}_{sha[:8]}.wav"
            out_segs.append({
                "wav_path": os.path.join(temp_dir, fname), "segment": seg, "sha": sha,
                "orig_file": orig_file, "duration_sec": dur
            })
    
    return out_segs


def preprocess_segment(seg_dict: Dict, min_snr_db: float = 10.0) -> Dict:
    """Führt ein robustes Audio-Preprocessing und Qualitätsprüfungen für ein Segment durch."""
    try:
        segment = seg_dict['segment']
        sr = segment.frame_rate
        samples = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0

        # =============== 1. Audio-Verarbeitung ===============

        # --- Rauschunterdrückung (Spektrale Subtraktion) ---
        stft = librosa.stft(samples, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        frame_power = np.mean(magnitude**2, axis=0)
        noise_threshold = np.percentile(frame_power, 15)
        noise_frames = frame_power < noise_threshold
        noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
        magnitude_denoised = np.maximum(magnitude - 1.2 * noise_profile, 0.0)
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        processed_samples = librosa.istft(stft_denoised, hop_length=512, length=len(samples))

        # --- Hochpassfilter ---
        nyquist = sr / 2
        cutoff = 80 / nyquist
        b, a = butter(4, cutoff, btype='high')
        processed_samples = filtfilt(b, a, processed_samples)

        # --- De-Essing ---
        stft_deess = librosa.stft(processed_samples, n_fft=2048, hop_length=512)
        mag_deess, phase_deess = np.abs(stft_deess), np.angle(stft_deess)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        deess_mask = np.ones_like(freqs)
        deess_range = (freqs >= 5500) & (freqs <= 9500)
        deess_mask[deess_range] = 0.5
        mag_deess = mag_deess * deess_mask[:, np.newaxis]
        stft_deessed = mag_deess * np.exp(1j * phase_deess)
        processed_samples = librosa.istft(stft_deessed, hop_length=512, length=len(processed_samples))

        # --- RMS-Normalisierung auf -23 dBFS ---
        rms = np.sqrt(np.mean(processed_samples**2))
        if rms > 1e-5:
            target_rms = 10 ** (-23 / 20)
            processed_samples = processed_samples * (target_rms / rms)

        # --- Soft Limiter ---
        threshold = 0.98
        processed_samples = np.tanh(processed_samples / threshold) * threshold

        # --- Stille am Anfang/Ende entfernen ---
        processed_samples, _ = librosa.effects.trim(processed_samples, top_db=40, frame_length=1024, hop_length=256)

        # =============== 2. Qualitätsprüfungen ===============
        quality_metrics = {}

        # --- SNR-Prüfung (verbessert) ---
        stft_qual = librosa.stft(processed_samples, n_fft=2048, hop_length=512)
        mag_qual = np.abs(stft_qual)
        frame_power_qual = np.mean(mag_qual**2, axis=0)
        noise_perc = np.percentile(frame_power_qual, 20)
        
        noise_power = np.mean(frame_power_qual[frame_power_qual < noise_perc])
        signal_power = np.mean(frame_power_qual[frame_power_qual >= noise_perc])
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10)) if signal_power > 0 and noise_power > 0 else 0.0
        quality_metrics['snr_db'] = float(snr_db)

        if snr_db < min_snr_db:
            seg_dict.update({'quality_rejected': True, 'reject_reason': f"Low SNR: {snr_db:.1f}dB"})
            return seg_dict

        # --- Clipping-Prüfung ---
        clipping_samples = np.sum(np.abs(processed_samples) >= 0.99)
        clipping_percent = (clipping_samples / len(processed_samples)) * 100 if len(processed_samples) > 0 else 0
        quality_metrics['clipping_percent'] = float(clipping_percent)

        if clipping_percent > 0.2:
            seg_dict.update({'quality_rejected': True, 'reject_reason': f"Clipping: {clipping_percent:.2f}%"})
            return seg_dict

        # --- Längenprüfung ---
        if len(processed_samples) < sr * 1.0:
            seg_dict.update({'quality_rejected': True, 'reject_reason': "Zu kurz nach Trimming"})
            return seg_dict

        # --- Prüfung auf interne Stille ---
        non_silent_intervals = librosa.effects.split(processed_samples, top_db=40, frame_length=1024, hop_length=256)
        speech_duration = sum(end - start for start, end in non_silent_intervals) / sr
        total_duration = len(processed_samples) / sr
        silence_ratio = (1 - (speech_duration / total_duration)) if total_duration > 0 else 0
        quality_metrics['silence_ratio'] = float(silence_ratio)

        if silence_ratio > 0.30:
            seg_dict.update({'quality_rejected': True, 'reject_reason': f"Zu viel interne Stille: {silence_ratio:.1%}"})
            return seg_dict

        # =============== 3. Abschluss ===============
        samples_int16 = (processed_samples * 32767).astype(np.int16)
        processed_segment = AudioSegment(data=samples_int16.tobytes(), sample_width=2, frame_rate=sr, channels=1)
        
        seg_dict.update({
            'segment': processed_segment,
            'quality_metrics': quality_metrics,
            'quality_rejected': False
        })
        
        return seg_dict
        
    except Exception as e:
        seg_dict.update({'quality_rejected': True, 'reject_reason': f"Processing error: {str(e)}"})
        return seg_dict