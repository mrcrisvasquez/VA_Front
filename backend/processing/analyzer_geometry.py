import numpy as np
from typing import Dict, Any, List

# Grid System (basado en Position Grid Pro)
# Valores normalizados de columnas (X)
GRID_X = [
    0.1875, # 360
    0.2500, # 480
    0.3125, # 600
    0.5000, # 960 (Centro)
    0.6875, # 1320
    0.7500, # 1440
    0.8125  # 1560
]

# Zonas Semánticas y Filas (Y)
GRID_ZONES = {
    "titulo": {
        "rows": [0.1667, 0.2500], # 180, 270
        "y_range": (0.0, 0.30)
    },
    "cuerpo": {
        "rows": [0.3333, 0.5000, 0.6667], # 360, 540, 720
        "y_range": (0.30, 0.75)
    },
    "disclaimer": {
        "rows": [0.7500, 0.8333], # 810, 900
        "y_range": (0.75, 1.0)
    }
}

GRID_TOLERANCE_X = 0.02 # 2% tolerancia en X
GRID_TOLERANCE_Y = 0.04 # 4% tolerancia en Y (un poco más flexible)

def get_semantic_zone(bbox: List[float]) -> str:
    """Identifica la zona semántica basada en la posición Y."""
    center_y = (bbox[0] + bbox[2]) / 2.0
    
    if center_y < 0.30:
        return "titulo"
    elif center_y > 0.75:
        return "disclaimer"
    return "cuerpo"

SAFE_ZONES = {
    "margin": 0.05        # Margen de seguridad en bordes
}

# Umbrales WCAG 2.2 para contraste
# https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html
WCAG_CONTRAST = {
    "normal_text": 4.5,   # Texto normal (< 18pt)
    "large_text": 3.0,    # Texto grande (>= 18pt o >= 14pt bold)
    "graphics": 3.0       # Elementos gráficos y UI
}

# Umbral para considerar texto "grande" en términos relativos
# Si la altura del texto ocupa más del 3% de la imagen, es texto grande
LARGE_TEXT_THRESHOLD = 0.03


def check_black_frame(frame: np.ndarray, timestamp: float, threshold: int = 10) -> Dict[str, Any]:
    """Detecta frames completamente negros."""
    if np.mean(frame) < threshold:
        return {"type": "black_frames", "timestamp": timestamp, "confidence": 1.0}
    return None


def get_luminance(bgr_color):
    """Calcula luminancia relativa según WCAG."""
    b, g, r = bgr_color
    color = [c / 255.0 for c in [r, g, b]]
    for i, c in enumerate(color):
        if c <= 0.03928: 
            color[i] = c / 12.92
        else: 
            color[i] = ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def is_large_text(bbox: List[float], frame_height: int) -> bool:
    """
    Determina si el texto es "grande" según WCAG.
    
    WCAG define texto grande como:
    - 18pt (24px) o mayor
    - 14pt (18.5px) bold o mayor
    
    Como no tenemos acceso al tamaño en puntos directamente,
    usamos la altura relativa del texto como aproximación.
    """
    # bbox: [ymin, xmin, ymax, xmax] en valores normalizados (0-1)
    text_height_normalized = bbox[2] - bbox[0]
    
    # En una imagen 1080p, 24px sería ~2.2% de la altura
    # Usamos 3% como umbral para ser más permisivos
    return text_height_normalized >= LARGE_TEXT_THRESHOLD


def check_contrast(frame: np.ndarray, bbox: List[float], text: str = "") -> Dict[str, Any]:
    """
    Verifica contraste según WCAG 2.2.
    
    Aplica diferentes umbrales según el tamaño del texto:
    - Texto grande (>= 3% altura): Mínimo 3:1
    - Texto normal (< 3% altura): Mínimo 4.5:1
    """
    try:
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = int(bbox[0]*h), int(bbox[1]*w), int(bbox[2]*h), int(bbox[3]*w)
        
        if xmax <= xmin or ymax <= ymin: 
            return None
        
        text_area = frame[ymin:ymax, xmin:xmax]
        if text_area.size == 0: 
            return None
        
        avg_text = np.mean(text_area, axis=(0, 1))
        
        # Muestrear fondo (borde exterior)
        pad = 5
        bg_samples = []
        if ymin-pad >= 0: 
            bg_samples.append(frame[ymin-pad:ymin, xmin:xmax])
        if ymax+pad <= h: 
            bg_samples.append(frame[ymax:ymax+pad, xmin:xmax])
        
        if not bg_samples: 
            return None
        
        bg_area = np.concatenate(bg_samples, axis=0)
        if bg_area.size == 0: 
            return None
        
        avg_bg = np.mean(bg_area, axis=(0, 1))
        
        l1 = get_luminance(avg_text)
        l2 = get_luminance(avg_bg)
        ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
        
        # Determinar umbral según tamaño del texto
        is_large = is_large_text(bbox, h)
        min_ratio = WCAG_CONTRAST["large_text"] if is_large else WCAG_CONTRAST["normal_text"]
        
        if ratio < min_ratio:
            size_label = "grande" if is_large else "normal"
            return {
                "type": "contrast",
                "issue": f"Contraste bajo ({ratio:.1f}:1, mín: {min_ratio}:1 para texto {size_label})",
                "ratio": round(ratio, 2),
                "required_ratio": min_ratio,
                "text_size": size_label,
                "confidence": 1.0
            }
    except Exception:
        return None
    return None


def check_grid_alignment(bbox: List[float], text_content: str = "") -> List[Dict[str, Any]]:
    """
    Verifica alineación estricta contra el Sistema de Grilla.
    """
    findings = []
    
    # bbox es [ymin, xmin, ymax, xmax]
    center_x = (bbox[1] + bbox[3]) / 2.0
    center_y = (bbox[0] + bbox[2]) / 2.0
    
    # 1. Validar Safe Zones (Bordes)
    margin = SAFE_ZONES["margin"]
    if bbox[1] < margin or bbox[3] > (1 - margin):
        findings.append({
            "type": "centering",
            "issue": "Texto fuera de márgenes seguros",
            "position_type": "safe_zone"
        })
        return findings # Si está fuera de margen, probablemente falle todo lo demás

    # 2. Identificar Zona Semántica
    zone_name = get_semantic_zone(bbox)
    zone_rules = GRID_ZONES[zone_name]
    
    # 3. Validar Grilla X (Columnas)
    # Debe coincidir con ALGUNA de las columnas definidas
    aligned_x = False
    closest_x_val = 0
    min_x_diff = 1.0
    
    for grid_x in GRID_X:
        diff = abs(center_x - grid_x)
        if diff < min_x_diff:
            min_x_diff = diff
            closest_x_val = grid_x
            
        if diff < GRID_TOLERANCE_X:
            aligned_x = True
            break
            
    if not aligned_x:
        findings.append({
            "type": "centering",
            "issue": f"Desalineado en horizontal (X). Sugerido: {closest_x_val:.2f}",
            "position_type": "grid_x"
        })

    # 4. Validar Grilla Y (Filas de la Zona)
    # Debe coincidir con alguna fila de SU zona semántica
    aligned_y = False
    closest_y_val = 0
    min_y_diff = 1.0
    
    for grid_y in zone_rules["rows"]:
        diff = abs(center_y - grid_y)
        if diff < min_y_diff:
            min_y_diff = diff
            closest_y_val = grid_y
            
        if diff < GRID_TOLERANCE_Y:
            aligned_y = True
            break
            
    if not aligned_y:
        # Solo reportar error de Y si está realmente lejos de las filas esperadas
        # A veces la altura del texto varía, así que somos permisivos si el error es pequeño
        if min_y_diff > 0.05:
            findings.append({
                "type": "centering",
                "issue": f"Posición vertical atípica para {zone_name}. Sugerido: {closest_y_val:.2f}",
                "position_type": "grid_y"
            })

    return findings


def process_frame_geometry(frame: np.ndarray, ocr_results: Dict[str, Any], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Orquestador de análisis geométrico.
    """
    findings = []
    detections = ocr_results.get("detections", [])
    
    if not detections:
        return findings
        
    analysis_types = settings.get("analysisTypes", [])
    
    for detection in detections:
        bbox = detection.get("bbox", [0, 0, 0, 0])
        text = detection.get("text", "")
        
        # Verificar contraste (con tamaño de texto)
        if "contrast" in analysis_types:
            contrast_result = check_contrast(frame, bbox, text)
            if contrast_result:
                contrast_result["text"] = text[:30]
                findings.append(contrast_result)
        
        # Verificar centrado/alineación
        # Verificar centrado/alineación (Sistema de Grilla)
        if "centering" in analysis_types:
            grid_results = check_grid_alignment(bbox, text)
            for result in grid_results:
                result["text"] = text[:30]
                findings.append(result)
    
    return findings