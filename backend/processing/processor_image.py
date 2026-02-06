import cv2
import os
import numpy as np
from typing import Dict, Any, List

# Importar los módulos de análisis
from . import analyzer_vision
from . import analyzer_geometry
from . import analyzer_synthesis

def process_image(image_path: str, settings: Dict[str, Any], ocr_model: Any, qwen_vl_model: Any = None) -> Dict[str, Any]:
    """
    Función orquestadora para una sola imagen.
    Se beneficia automáticamente del pre-procesamiento (CLAHE) y Rescate (Upscaling)
    implementado en analyzer_vision.
    """
    print(f"[ImageProcessor] Analizando imagen: {image_path}")
    
    try:
        # 1. Cargar la imagen
        frame = cv2.imread(image_path)
        if frame is None:
            raise IOError(f"No se pudo leer la imagen: {image_path}")

        # 2. Configuración de exclusiones y análisis forzado
        excluded_words_str = settings.get("excludedWords", "")
        excluded_words = excluded_words_str.split(",") if excluded_words_str else []

        # Si el Visual Quantifier está activo, forzamos el análisis técnico (Contraste, Alineación)
        # para nutrir el contexto del LLM (Qwen).
        if settings.get("enableQuantifier", False):
            current_types = set(settings.get("analysisTypes", []))
            current_types.update(["contrast", "centering", "ocr"])
            settings["analysisTypes"] = list(current_types)
            print("[ImageProcessor] Análisis técnico forzado por Quantifier.")
        
        # 3. OCR con EasyOCR (incluye CLAHE + Upscaling interno)
        ocr_results = analyzer_vision.get_text_and_boxes_from_frame(
            frame=frame,
            ocr_model=ocr_model, 
            excluded_words=excluded_words
        )
        
        # 4. Analizar Geometría
        geometry_report = analyzer_geometry.process_frame_geometry(
            frame=frame,
            ocr_results=ocr_results,
            settings=settings 
        )
        
        # 5. Clasificar hallazgos
        ocr_issues = ocr_results.get("ocr_issues", [])
        centering_issues = []
        contrast_issues = []
        
        for issue in geometry_report:
            issue_type = issue.get("type")
            if issue_type == "centering":
                centering_issues.append(issue)
            elif issue_type == "contrast":
                contrast_issues.append(issue)
            else:
                ocr_issues.append(issue)
        
        final_report = {
            "filename": os.path.basename(image_path),
            "findings": {
                "ocr": ocr_issues,
                "centering": centering_issues,
                "contrast": contrast_issues
            }
        }
        
        # 6. (NUEVO) Visual Quantifier 5-Axis
        # Solo si está habilitado en settings y tenemos el modelo
        if settings.get("enableQuantifier", False) and qwen_vl_model:
             # Generar contexto técnico para "Ground Truth"
             tech_context = []
             
             if contrast_issues:
                 tech_context.append(f"- Errores de Contraste (WCAG): {len(contrast_issues)} (CRÍTICO: Colores ilegibles detectados)")
             
             if centering_issues:
                 tech_context.append(f"- Problemas de Alineación: {len(centering_issues)} (Texto no centrado o desalineado)")
             
             # Verificar densidad de texto (Diseños vacíos)
             full_text = ocr_results.get("text", [])
             word_count = sum(len(t.split()) for t in full_text)
             if word_count < 3:
                 tech_context.append("- Densidad de Texto: Muy baja (< 3 palabras). Posible falta de contenido.")

             tech_context_str = "\n".join(tech_context)
             
             quantifier_data = analyzer_synthesis.analyze_image_quantifier(image_path, qwen_vl_model, technical_context=tech_context_str)
             final_report["quantifier_data"] = quantifier_data
        
        print(f"[ImageProcessor] Análisis completado: {image_path}")
        return final_report
        
    except Exception as e:
        print(f"[ImageProcessor] ERROR al procesar {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "filename": os.path.basename(image_path),
            "error": str(e)
        }