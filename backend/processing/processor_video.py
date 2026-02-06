import os
import tempfile
import ffmpeg
import cv2
import json
from typing import Dict, Any, List
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque

# Importar nuestros módulos de análisis
from . import analyzer_audio
from . import analyzer_vision
from . import analyzer_geometry
from . import analyzer_synthesis

# --- Extracción de Medios ---

def extract_audio_from_video(video_path: str) -> str:
    """Extrae el audio a un archivo WAV temporal."""
    print(f"[VideoProcessor] Extrayendo audio de: {video_path}")
    try:
        probe = ffmpeg.probe(video_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream is None:
            return None 
    except ffmpeg.Error as e:
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        try:
            (
                ffmpeg
                .input(video_path)
                .output(tmp_audio.name, acodec='pcm_s16le', ac=1, ar='16000') 
                .run(quiet=True, overwrite_output=True)
            )
            return tmp_audio.name
        except ffmpeg.Error:
            if os.path.exists(tmp_audio.name): os.remove(tmp_audio.name)
            return None

def extract_smart_frames(video_path: str, depth: str, similarity_threshold=0.98) -> List[Dict[str, Any]]:
    """Extrae frames clave basados en cambios visuales (SSIM)."""
    print(f"[VideoProcessor] Extrayendo frames (Profundidad: {depth})")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se puede abrir el video: {video_path}")

    frames_to_analyze = []
    last_keyframe = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    force_interval = int(fps * 2) 
    
    if depth == "lite": frame_interval = int(fps * 2)
    elif depth == "normal": frame_interval = int(fps * 0.5)
    else: frame_interval = 1 

    while True:
        ret, frame = cap.read()
        if not ret: break
        timestamp = frame_count / fps
        should_save = False
        
        if depth != "full":
            if frame_count % frame_interval == 0: should_save = True
        else:
            if frame_count % frame_interval == 0:
                if last_keyframe is None: should_save = True
                else:
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sim = ssim(last_keyframe, current_gray, data_range=255)
                    if np.isnan(sim): sim = 1.0
                    if sim < similarity_threshold or frame_count % force_interval == 0:
                        should_save = True
                        
        if should_save:
            frames_to_analyze.append({"timestamp": timestamp, "frame": frame})
            last_keyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_count += 1
    cap.release()
    return frames_to_analyze


def consolidate_issues(issues: List[Dict[str, Any]], key_field: str, fps_interval: float = 0.5) -> List[Dict[str, Any]]:
    """
    Agrupa errores consecutivos del mismo tipo/texto en un solo reporte con duración.
    Similar a cómo se agrupan silencios en el análisis de audio.
    
    Args:
        issues: Lista de hallazgos con 'timestamp' y campo clave
        key_field: Campo para agrupar (ej: 'text' para OCR, 'issue' para geometría)
        fps_interval: Intervalo de tiempo entre frames analizados
    """
    if not issues:
        return []
    
    # 1. Agrupar issues por su clave única primero (para manejar intercalados)
    issues_by_key = {}
    for issue in issues:
        # Clave robusta: strip, lowercase.
        # Para OCR: texto.
        # Para Centering: issue msg.
        # Para Contrast: necesitamos limpiar la ratio dinámica.
        raw_key = issue.get(key_field, '')
        
        # Limpieza específica para Contraste (eliminar "2.1:1", etc. para agrupar)
        if "contrast" in issue.get('type', ''):
             # Ejemplo: "Contraste bajo (2.1:1, mín: 4.5:1...)"
             # Usamos solo la parte estática del mensaje o el tipo
             clean_key = raw_key.split('(')[0].strip() # "Contraste bajo"
        else:
             clean_key = str(raw_key).strip()
             
        if clean_key not in issues_by_key:
            issues_by_key[clean_key] = []
        issues_by_key[clean_key].append(issue)
        
    consolidated_all = []
    
    # 2. Consolidar temporalmente CADA grupo por separado
    for key, group_issues in issues_by_key.items():
        # Ordenar por tiempo
        sorted_group = sorted(group_issues, key=lambda x: x.get('timestamp', 0))
        
        current_block = None
        
        for issue in sorted_group:
            ts = issue.get('timestamp', 0)
            
            # Chequear continuidad con bloque actual
            if current_block:
                time_gap = ts - current_block['end_time']
                if time_gap <= 2.0: # Gap de 2s máximo para continuidad visual
                    current_block['end_time'] = ts
                    current_block['count'] += 1
                    # Actualizar contexto si es mejor?
                    continue
                else:
                    # Cerrar bloque anterior
                    duration = current_block['end_time'] - current_block['start_time'] + fps_interval
                    consolidated_all.append({
                        **current_block['data'], # Copiar datos base
                        'timestamp': current_block['start_time'],
                        'duration': round(duration, 2),
                        'occurrences': current_block['count']
                    })
                    current_block = None
            
            # Iniciar nuevo bloque
            if not current_block:
                current_block = {
                    'start_time': ts,
                    'end_time': ts,
                    'count': 1,
                    'data': issue # Guardar objeto completo como base
                }
                
        # Cerrar último bloque del grupo
        if current_block:
            duration = current_block['end_time'] - current_block['start_time'] + fps_interval
            consolidated_all.append({
                **current_block['data'],
                'timestamp': current_block['start_time'],
                'duration': round(duration, 2),
                'occurrences': current_block['count']
            })

    # 3. Reordenar todo por aparición temporal
    return sorted(consolidated_all, key=lambda x: x['timestamp'])


# --- Orquestador Principal ---

def process_video(video_path: str, settings: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    tmp_audio_path = None
    audio_report = {} 
    whisper_segments = [] 
    srt_content = "" 
    
    try:
        # 1. Audio
        tmp_audio_path = extract_audio_from_video(video_path)
        if tmp_audio_path:
            audio_report, whisper_segments, srt_content = analyzer_audio.process_audio(
                audio_path=tmp_audio_path, settings=settings,
                whisper_model=models["whisper"], vad_model=models["silero_vad"] 
            )

        # 2. Frames
        frames_to_analyze = extract_smart_frames(video_path, settings["depth"])
        
        print(f"[VideoProcessor] Analizando {len(frames_to_analyze)} frames con EasyOCR...")
        
        # Listas temporales (antes de consolidar)
        raw_ocr_issues = []
        raw_centering_issues = []
        raw_contrast_issues = []
        vision_black_frame_report = []
        all_ocr_text_by_time = [] 

        # Buffer de estabilidad para filtrar falsos positivos
        stability_buffer = deque(maxlen=3) 
        
        # Calcular intervalo entre frames
        fps_interval = 0.5  # Default para 'normal'
        if settings["depth"] == "lite":
            fps_interval = 2.0
        elif settings["depth"] == "full":
            fps_interval = 0.1
        
        for item in frames_to_analyze:
            frame = item["frame"]
            timestamp = item["timestamp"]
            
            # A. OCR y Detección
            ocr_results = analyzer_vision.get_text_and_boxes_from_frame(
                frame=frame,
                ocr_model=models["ocr"], 
                excluded_words=settings.get("excludedWords", "").split(",")
            )
            
            # Actualizar Buffer de Estabilidad
            current_tokens = set(ocr_results["raw_text"].lower().split())
            stability_buffer.append(current_tokens)
            
            # Calcular Intersección Estable
            if len(stability_buffer) > 0:
                stable_tokens = set.intersection(*stability_buffer)
            else:
                stable_tokens = current_tokens

            # B. Geometría
            geometry_report = analyzer_geometry.process_frame_geometry(
                frame=frame, ocr_results=ocr_results, settings=settings
            )
            
            # C. Frames Negros (gated by toggle)
            if "black_frames" in settings.get("analysisTypes", []):
                black_frame = analyzer_geometry.check_black_frame(frame, timestamp)
                if black_frame:
                    vision_black_frame_report.append(black_frame)
            
            # --- Recolectar hallazgos (sin consolidar aún) ---
            
            # Texto para Qwen
            if ocr_results.get("raw_text"):
                all_ocr_text_by_time.append({"time": timestamp, "text": ocr_results["raw_text"]})
            
            # Errores OCR (filtrado por estabilidad, gated by toggle)
            if "ocr" in settings.get("analysisTypes", []):
                if ocr_results.get("ocr_issues"):
                    for issue in ocr_results["ocr_issues"]:
                        word_check = issue['text'].lower()
                        if len(stability_buffer) < 2 or (word_check in stable_tokens):
                            raw_ocr_issues.append({"timestamp": timestamp, **issue})

            # Geometría (contraste y centrado, gated individually)
            for geo_issue in geometry_report:
                issue_with_ts = {"timestamp": timestamp, **geo_issue}
                if geo_issue.get("type") == "centering" and "centering" in settings.get("analysisTypes", []):
                    raw_centering_issues.append(issue_with_ts)
                elif geo_issue.get("type") == "contrast" and "contrast" in settings.get("analysisTypes", []):
                    raw_contrast_issues.append(issue_with_ts)

        # --- CONSOLIDAR ERRORES (agrupar por palabra/tipo con duración) ---
        print("[VideoProcessor] Consolidando errores duplicados...")
        
        vision_ocr_report = consolidate_issues(raw_ocr_issues, 'text', fps_interval)
        vision_centering_report = consolidate_issues(raw_centering_issues, 'issue', fps_interval)
        vision_contrast_report = consolidate_issues(raw_contrast_issues, 'issue', fps_interval)
        
        # Frames negros también se consolidan
        vision_black_frame_report = consolidate_issues(vision_black_frame_report, 'type', fps_interval)

        print(f"[VideoProcessor] OCR: {len(raw_ocr_issues)} -> {len(vision_ocr_report)} (consolidados)")
        print(f"[VideoProcessor] Centrado: {len(raw_centering_issues)} -> {len(vision_centering_report)} (consolidados)")
        print(f"[VideoProcessor] Contraste: {len(raw_contrast_issues)} -> {len(vision_contrast_report)} (consolidados)")

        # 3. Síntesis (Qwen) — gated by toggle
        synthesis_report = []
        synthesis_log_entries = []

        if "synthesis" in settings.get("analysisTypes", []):
            print(f"[VideoProcessor] Iniciando síntesis multimodal con Qwen...")

            # Calcular duración total
            total_duration = 0
            if whisper_segments: total_duration = max(total_duration, whisper_segments[-1]['end'] / 1000.0)
            if all_ocr_text_by_time: total_duration = max(total_duration, all_ocr_text_by_time[-1]['time'])

            CHUNK_DURATION = 60
            for start_time in range(0, int(total_duration) + 1, CHUNK_DURATION):
                end_time = start_time + CHUNK_DURATION

                audio_chunk = [s for s in whisper_segments if s['start'] >= (start_time*1000) and s['end'] < (end_time*1000)]
                ocr_chunk = [o for o in all_ocr_text_by_time if o['time'] >= start_time and o['time'] < end_time]
                silence_chunk = [s for s in audio_report.get("silences", []) if s['start'] >= start_time and s['start'] < end_time]

                if not audio_chunk and not ocr_chunk and not silence_chunk: continue

                chunk_report = analyzer_synthesis.compare_audio_to_video(
                    whisper_segments=audio_chunk, ocr_data=ocr_chunk,
                    silence_data=silence_chunk, qwen_model=models["qwen_vl"]
                )
                synthesis_report.extend(chunk_report)
        else:
            print("[VideoProcessor] Síntesis desactivada por usuario.")

        # Compile synthesis log entry
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "video",
            "details": {
                "audio_segments": len(whisper_segments),
                "ocr_texts": len(all_ocr_text_by_time),
                "frames_analyzed": len(frames_to_analyze),
                "issues": {
                    "ocr": len(vision_ocr_report),
                    "centering": len(vision_centering_report),
                    "contrast": len(vision_contrast_report),
                    "black_frames": len(vision_black_frame_report),
                    "synthesis": len(synthesis_report)
                }
            },
            "synthesis_findings": synthesis_report
        }
        synthesis_log_entries.append(log_entry)

        return {
            "report": {
                **audio_report,
                "ocr": vision_ocr_report,
                "centering": vision_centering_report,
                "black_frames": vision_black_frame_report,
                "contrast": vision_contrast_report,
                "synthesis": synthesis_report
            },
            "srt": srt_content,
            "synthesis_log": synthesis_log_entries
        }

    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)