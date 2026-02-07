import json
from typing import Dict, Any, List
from . import gemini_client

# --- PROMPT MEJORADO PARA QWEN2.5 ---
# Más directo, sin system message elaborado, instrucciones claras
SYNTHESIS_PROMPT_TEMPLATE = """Analiza los siguientes datos y encuentra DISCREPANCIAS entre lo que se DICE (audio) y lo que se VE (texto en pantalla).

AUDIO TRANSCRITO:
{audio_segments_str}

SILENCIOS DETECTADOS:
{silence_data}

TEXTO EN PANTALLA (OCR):
{ocr_data}

INSTRUCCIONES:
1. Busca datos específicos (números, precios, fechas, nombres) que sean DIFERENTES entre audio y pantalla.
2. Si el audio habla de algo pero la pantalla está vacía, es una discrepancia.
3. Si audio y pantalla coinciden, o no hay datos comparables, responde: CONSISTENTE

FORMATO DE RESPUESTA (solo usar si hay problemas):
- [Discrepancia] (MM:SS): Descripción del problema.

Responde SOLO con la lista de discrepancias o la palabra CONSISTENTE:"""


def format_ocr_data_for_prompt(ocr_data: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Formatea la lista de OCR para el prompt."""
    if not ocr_data:
        return "No se detectó texto en pantalla."
    
    lines = []
    total_chars = 0
    for item in ocr_data:
        time_str = f"{int(item['time'] // 60):02d}:{int(item['time'] % 60):02d}"
        line = f"- ({time_str}) \"{item['text']}\""
        if total_chars + len(line) > max_chars:
            lines.append("... (truncado)")
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines)


def format_audio_segments_for_prompt(whisper_segments: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Formatea el audio para el prompt."""
    if not whisper_segments:
        return "No se detectó voz (silencio o música de fondo)."
    
    full_text = "\n".join([s['text'].strip() for s in whisper_segments])
    if len(full_text) > max_chars:
        return full_text[:max_chars] + "\n... (truncado)"
    return full_text


def format_silence_data_for_prompt(silence_data: List[Dict[str, Any]]) -> str:
    """Formatea los datos de silencio."""
    if not silence_data:
        return "Sin silencios significativos."
    lines = []
    for item in silence_data:
        lines.append(f"- Silencio de {item['duration']:.1f}s en {item['start']:.1f}s")
    return "\n".join(lines)


def compare_audio_to_video(
    whisper_segments: List[Dict[str, Any]],
    ocr_data: List[Dict[str, Any]],
    silence_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Usa Gemini para comparar audio y video."""

    if not whisper_segments and not ocr_data and not silence_data:
        return []

    print("[SynthesisAnalyzer] Ejecutando inferencia de síntesis...")

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        audio_segments_str=format_audio_segments_for_prompt(whisper_segments),
        silence_data=format_silence_data_for_prompt(silence_data),
        ocr_data=format_ocr_data_for_prompt(ocr_data)
    )

    try:
        response_text = gemini_client.synthesis_compare(prompt)
        print(f"[Synthesis] Respuesta Gemini: {response_text[:150]}...")
        
        findings = []
        
        # Detectar si el modelo ignoró las instrucciones
        ignore_phrases = ["Hello", "How can I", "I'm here to", "Hola", "¿En qué puedo"]
        if any(phrase in response_text for phrase in ignore_phrases):
            print("[Synthesis] ADVERTENCIA: Modelo respondió como chatbot. Ignorando respuesta.")
            return []

        # Parser de respuesta
        response_upper = response_text.upper()
        
        # Si es consistente, no hay problemas
        if "CONSISTENTE" in response_upper and len(response_text) < 100:
            return [] 
        
        # Buscar discrepancias reportadas
        if "[DISCREPANCIA]" in response_upper or "DISCREPANCIA" in response_upper:
            lines = response_text.split('\n')
            for line in lines:
                line_clean = line.strip()
                if not line_clean:
                    continue
                # Buscar líneas que parecen reportes de discrepancia
                if any(marker in line_clean.upper() for marker in ["DISCREPANCIA", "[DISCREPANCIA]", "- ["]):
                    # Limpiar el texto
                    issue_text = line_clean
                    for remove in ["- [Discrepancia]", "[Discrepancia]", "- ", "[", "]"]:
                        issue_text = issue_text.replace(remove, "")
                    issue_text = issue_text.strip()
                    
                    if issue_text and len(issue_text) > 5:
                        findings.append({
                            "type": "synthesis",
                            "issue": issue_text
                        })
        
        # Si hay contenido sustancial pero sin tags, reportar como observación
        elif len(response_text) > 20 and "CONSISTENTE" not in response_upper:
            # Evitar respuestas muy largas o que parezcan explicaciones
            if len(response_text) < 300:
                findings.append({
                    "type": "synthesis",
                    "issue": f"Observación: {response_text[:200]}"
                })

        return findings

    except Exception as e:
        print(f"[SynthesisAnalyzer] Error en Gemini: {e}")
        return []


# --- VISUAL QUANTIFIER (5-AXIS) ---

QUANTIFIER_SYSTEM_PROMPT = """Eres un Analista Visual Senior. Tu trabajo es evaluar objetivamente la calidad técnica y estética de la imagen. No seas excesivamente negativo ni inventes fallos. Basa tus puntuaciones en la evidencia visible."""

QUANTIFIER_USER_PROMPT = """### PARÁMETROS DE EVALUACIÓN (Escala 1 a 10)
Evalúa con justicia. Usa toda la escala (5 es promedio, 8 es muy bueno).

1. MODERNO (Tendencia)
- Evalúa: Estética actual y limpieza visual.
- Puntuación: 1-3 (Anticuado/Sucio) | 4-6 (Promedio/Estándar) | 7-10 (Tendencia/Premium).

2. ENSEÑA (Claridad Informativa)
- Evalúa: ¿Se entiende el mensaje rápido? Jerarquía clara.
- Penaliza solo si es confuso o ilegible.

3. BUEN DISEÑO (Técnica Gestalt)
- Evalúa: Alineación, márgenes y equilibrio.
- INSTRUCCIÓN: Si recibes un reporte de errores técnicos, úsalo como guía, pero evalúa la composición global.

4. HUMANO (Conexión Emocional)
- BUSCA ACTIVAMENTE: Personas, rostros, manos, partes del cuerpo.
- TABLA DE PUNTAJE:
  * 1-2: Absolutamente NADA humano (solo texto/formas).
  * 3-5: Manos, siluetas, o rostros pequeños/lejanos.
  * 6-8: Retratos claros, personas interactuando, fotografía de stock buena.
  * 9-10: Fotografía auténtica, emotiva y muy expresiva.
- IMPORTANTE: Si ves una persona o parte de una persona, el score MÍNIMO es 4.

5. TIPOGRÁFICO (Protagonismo de la letra)
- Evalúa: Calidad de fuentes y legibilidad.
- Penaliza: Deformaciones o falta de legibilidad.

### INSTRUCCIONES DE SALIDA (FORMATO JSON ÚNICAMENTE)
Analiza la imagen y devuelve UNICAMENTE un objeto JSON válido.
El JSON debe tener esta estructura exacta:

{
  "analisis": {
    "moderno": {
      "score": (1-10),
      "razon": "Justificación objetiva."
    },
    "ensena": {
      "score": (1-10),
      "razon": "Justificación objetiva."
    },
    "buen_diseno": {
      "score": (1-10),
      "razon": "Justificación objetiva."
    },
    "humano": {
      "score": (1-10),
      "razon": "Describir qué elementos humanos se ven (o confirmar su ausencia)."
    },
    "tipografico": {
      "score": (1-10),
      "razon": "Justificación objetiva."
    }
  },
  "score_promedio": (decimal),
  "conclusion": "Resumen objetivo."
}"""

def analyze_image_quantifier(image_path: str, technical_context: str = "") -> Dict[str, Any]:
    """
    Ejecuta el análisis visual de 5 ejes utilizando Gemini 2.0 Flash.
    Inyecta contexto técnico (errores) para ajustar la puntuación.
    """
    print(f"[Quantifier] Iniciando análisis visual 5-Ejes para: {image_path}")

    try:
        # Inyección de Contexto
        final_prompt = QUANTIFIER_USER_PROMPT
        if technical_context:
            final_prompt = f"""### REPORTE TÉCNICO PREVIO (GROUND TRUTH)
La siguiente información proviene de algoritmos de detección precisos. NO la ignores.
{technical_context}

INSTRUCCIÓN CRÍTICA:
Si el reporte técnico indica errores de Contraste o Alineación, DEBES penalizar severamente los puntajes de 'Buen Diseño' y 'Enseña'.
- 1-3 errores graves: Máximo 7/10 en Buen Diseño.
- >3 errores graves: Máximo 5/10 en Buen Diseño.

{QUANTIFIER_USER_PROMPT}"""

        data = gemini_client.quantifier_analyze(image_path, QUANTIFIER_SYSTEM_PROMPT, final_prompt)
        print(f"[Quantifier] Análisis completado.")
        return data

    except Exception as e:
        print(f"[Quantifier] Error generando reporte: {e}")
        return {"error": str(e)}