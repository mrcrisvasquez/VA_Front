import cv2
import numpy as np
import os
from typing import Dict, Any, List
from spellchecker import SpellChecker
from . import gemini_client

def normalize_box(box_points: List[List[float]], img_width: int, img_height: int) -> List[float]:
    """Convierte bbox absoluta a relativa [ymin, xmin, ymax, xmax]."""
    try:
        xs = [pt[0] for pt in box_points]
        ys = [pt[1] for pt in box_points]
        return [
            max(0.0, min(1.0, min(ys) / img_height)), # ymin
            max(0.0, min(1.0, min(xs) / img_width)),  # xmin
            max(0.0, min(1.0, max(ys) / img_height)), # ymax
            max(0.0, min(1.0, max(xs) / img_width))   # xmax
        ]
    except Exception:
        return [0,0,0,0]

def get_text_and_boxes_from_frame(frame: np.ndarray, excluded_words: List[str]) -> Dict[str, Any]:
    """
    Analiza un frame con Gemini 2.0 Flash OCR.
    Gemini handles preprocessing internally, so no CLAHE/upscale strategies needed.
    """
    height, width = frame.shape[:2]
    lines_found = []

    try:
        lines_found = gemini_client.ocr_frame(frame)
        if lines_found:
            print(f"[Vision] Gemini OCR detectó {len(lines_found)} elementos.")
    except Exception as e:
        print(f"[Vision] Error en Gemini OCR: {e}")

    # Procesamiento de resultados
    ocr_results = []
    ocr_issues = []
    raw_text_list = []
    
    # Spellchecker bilingüe: español + inglés (para contenido tech)
    # Cargar diccionario personalizado si existe para mejorar cobertura (corrección para 'eres', 'seguidas', etc.)
    spell_es = SpellChecker(language='es')
    
    try:
        custom_dict_path = os.path.join(os.path.dirname(__file__), "..", "data", "es_50k.txt")
        if os.path.exists(custom_dict_path):
            # Formato archivo: "palabra frecuencia"
            custom_words = []
            with open(custom_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        custom_words.append(parts[0]) # Solo la palabra
            
            spell_es.word_frequency.load_words(custom_words)
            print(f"[Vision] Diccionario ES extendido cargado ({len(custom_words)} palabras extra).")
    except Exception as e:
        print(f"[Vision] Advertencia: No se pudo cargar diccionario extendido: {e}")

    spell_en = SpellChecker(language='en')
    
    # Normalizar lista de excluidos
    excluded_set = set(w.lower().strip() for w in excluded_words if w.strip())

    for i, detection in enumerate(lines_found):
        try:
            # EasyOCR devuelve: (bbox_points, text, confidence)
            # bbox_points es lista de 4 puntos [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            coords = detection[0]
            text = detection[1]
            conf = detection[2]

            text = text.strip()
            
            # Filtro de confianza bajo (0.1 permite detectar textos difíciles, el spellchecker filtra después)
            if not text or conf < 0.1: 
                continue 
            
            raw_text_list.append(text)
            
            # Normalizar caja (usa width/height actualizados si hubo resize)
            norm_box = normalize_box(coords, width, height)
            
            ocr_results.append({
                "text": text,
                "bbox": norm_box,
                "confidence": conf
            })
            
            # Verificación Ortográfica Bilingüe
            clean_text_for_spell = ''.join(c for c in text if c.isalnum() or c.isspace())
            for word in clean_text_for_spell.split():
                # Reglas: >2 letras (incluye palabras de 3 letras), sin números, no está en excluidos
                if len(word) > 2 and not any(char.isdigit() for char in word):
                    word_lower = word.lower()
                    # Excluir palabras de la lista del usuario
                    if word_lower in excluded_set:
                        continue
                    
                    # Verificar en ambos idiomas (español e inglés)
                    # Una palabra es correcta si está en CUALQUIERA de los dos idiomas
                    is_valid_es = word_lower in spell_es
                    is_valid_en = word_lower in spell_en
                    
                    if not is_valid_es and not is_valid_en:
                        print(f"[DEBUG Spell] ERROR DETECTADO (No en diccionarios): {word}")
                        ocr_issues.append({
                            "type": "ocr",
                            "text": word,
                            "context": text,
                            "confidence": conf
                        })
                    
        except Exception as e:
            print(f"[VisionAnalyzer] Error procesando línea {i}: {e}")
            continue

    return {
        "detections": ocr_results,
        "ocr_issues": ocr_issues,
        "raw_text": " ".join(raw_text_list)
    }