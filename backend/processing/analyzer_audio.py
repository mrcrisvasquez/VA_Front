import os
import pydub
import torch
import numpy as np
import whisper
from typing import Dict, Any, List, Tuple

# Palabras clave a buscar
KEYWORDS_TO_FIND = ["pausa", "corte", "inicio", "grabando"]
# Muletillas comunes
FILLERS_TO_FIND = ["eh", "ehh", "ehhh", "um", "umm", "este", "o sea", "pues", "bueno"]

def get_speech_timestamps(audio_path: str, vad_model: Any) -> List[Dict[str, int]]:
    """
    Usa Silero VAD para encontrar segmentos de voz.

    NOTA: Esta función es un stub reservado para uso futuro. Está pensada para
    chunking de audios muy largos (>30 min) donde Whisper necesitaría procesar
    por segmentos de voz en lugar del archivo completo.

    Silero VAD se mantiene cargado en main.py para este propósito futuro.
    La detección de silencios actual usa pydub.silence.detect_silence() en
    find_silences(), que es suficiente para el análisis de QA.

    La llamada a esta función está comentada en process_audio() línea 108.
    """
    print("[AudioAnalyzer] Ejecutando VAD (simulado, procesando todo)")
    return [{"start": 0, "end": -1}]

def run_whisper(audio_path: str, whisper_model: Any, language: str) -> List[Dict[str, Any]]:
    """Ejecuta OpenAI-Whisper en el archivo de audio."""
    print(f"[AudioAnalyzer] Ejecutando OpenAI-Whisper (lang: {language})...")

    # 'whisper_model' es el modelo ya cargado
    result = whisper_model.transcribe(
        audio_path, 
        language=language if language != "auto" else None, 
        fp16=False # fp16 no es compatible con MPS, usamos fp32
    )

    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": int(segment["start"] * 1000), # Convertir segundos a milisegundos
            "end": int(segment["end"] * 1000),     # Convertir segundos a milisegundos
            "text": segment["text"]
        })

    print("[AudioAnalyzer] Whisper completado.")
    return segments

def format_time_srt(milliseconds: int) -> str:
    """Convierte ms a formato de tiempo SRT (HH:MM:SS,ms)"""
    sec = milliseconds / 1000
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace('.', ',')

def generate_srt(segments: List[Dict[str, Any]]) -> str:
    """Genera un string de formato SRT a partir de los segmentos de Whisper."""
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_time_srt(segment["start"])
        end_time = format_time_srt(segment["end"])
        text = segment["text"].strip()
        
        srt_content.append(str(i + 1))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text + "\n")
        
    return "\n".join(srt_content)

def find_silences(audio_path: str, silence_thresh_db: float = -35.0, min_silence_len_ms: int = 1000) -> List[Dict[str, Any]]:
    """Usa Pydub para encontrar silencios por debajo de un umbral de dB."""
    print(f"[AudioAnalyzer] Buscando silencios < {silence_thresh_db}dB")
    audio = pydub.AudioSegment.from_wav(audio_path)
    
    silence_chunks = pydub.silence.detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db
    )
    
    findings = []
    for start, end in silence_chunks:
        findings.append({
            "start": start / 1000.0, # Convertir a segundos
            "duration": (end - start) / 1000.0
        })
    return findings

def find_transcript_issues(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Busca palabras clave y muletillas en los segmentos transcritos."""
    print("[AudioAnalyzer] Buscando palabras clave y muletillas...")
    keywords_found = []
    fillers_found = []
    
    for segment in segments:
        text_lower = segment["text"].lower()
        start_sec = segment["start"] / 1000.0
        
        for keyword in KEYWORDS_TO_FIND:
            if f" {keyword} " in text_lower or text_lower.startswith(keyword):
                keywords_found.append({"start": start_sec, "keyword": keyword})
                
        for filler in FILLERS_TO_FIND:
            if f" {filler} " in text_lower or text_lower.startswith(filler):
                fillers_found.append({"start": start_sec, "text": filler})
                
    return keywords_found, fillers_found

def process_audio(audio_path: str, settings: Dict[str, Any], whisper_model: Any, vad_model: Any) -> Tuple[Dict[str, List], str, str]:
    """Orquestador principal para todas las tareas de audio."""
    
    # 1. Ejecutar VAD (actualmente simulado para procesar todo)
    # speech_chunks = get_speech_timestamps(audio_path, vad_model)
    
    # 2. Ejecutar Whisper
    # El 90% es ES/EN, Whisper `large-v3` detecta el idioma automáticamente
    whisper_segments = run_whisper(audio_path, whisper_model, language="auto")
    
    # 3. Generar transcripción completa y SRT
    full_transcript = " ".join([s["text"] for s in whisper_segments])
    srt_content = generate_srt(whisper_segments)
    
    report = {}
    
    # 4. Ejecutar análisis de silencio (Pydub)
    if "silences" in settings["analysisTypes"]:
        report["silences"] = find_silences(audio_path, silence_thresh_db=-35.0)
        
    # 5. Ejecutar análisis de transcripción (Palabras clave, Muletillas)
    keywords, fillers = find_transcript_issues(whisper_segments)
    if "keywords" in settings["analysisTypes"]:
        report["keywords"] = keywords
    if "fillers" in settings["analysisTypes"]:
        report["fillers"] = fillers
        
    return report, whisper_segments, srt_content