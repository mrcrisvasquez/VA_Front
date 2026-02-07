import os
import json
import base64
import tempfile
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from google import genai

_client = None

MODEL = "gemini-2.0-flash"


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def transcribe_audio(audio_path: str, language: str = "auto") -> List[Dict[str, Any]]:
    """
    Transcribe audio using Gemini 2.0 Flash.
    Returns segments in same format as Whisper: [{start: ms, end: ms, text: str}]
    """
    print(f"[GeminiClient] Transcribing audio: {audio_path}")
    client = get_client()

    lang_hint = ""
    if language and language != "auto":
        lang_hint = f" The audio is in {language}."

    uploaded_file = client.files.upload(file=audio_path)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            uploaded_file,
            f"Transcribe this audio with timestamps.{lang_hint} "
            "Return ONLY a JSON array with objects containing: "
            '"start" (milliseconds), "end" (milliseconds), "text" (string). '
            "Include timestamps for each sentence or phrase. "
            "Example: [{\"start\": 0, \"end\": 2500, \"text\": \"Hello world\"}]"
        ]
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    segments = json.loads(text)
    print(f"[GeminiClient] Transcription complete: {len(segments)} segments")
    return segments


def ocr_frame(frame: np.ndarray) -> List[Tuple]:
    """
    OCR a frame using Gemini 2.0 Flash.
    Returns results in EasyOCR format: [(bbox_points, text, confidence)]
    where bbox_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    client = get_client()

    height, width = frame.shape[:2]

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    b64_image = base64.b64encode(buffer).decode('utf-8')

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image
                        }
                    },
                    {
                        "text": (
                            f"Extract ALL visible text from this image ({width}x{height} pixels). "
                            "Return ONLY a JSON array. Each element must have: "
                            '"text" (the detected string), '
                            '"bbox" (array of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in pixel coordinates), '
                            '"confidence" (float 0-1). '
                            "If no text is visible, return an empty array []. "
                            "Example: [{\"text\": \"Hello\", \"bbox\": [[10,10],[100,10],[100,40],[10,40]], \"confidence\": 0.95}]"
                        )
                    }
                ]
            }
        ]
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    detections = json.loads(text)

    # Convert to EasyOCR format: (bbox_points, text, confidence)
    results = []
    for det in detections:
        bbox = det.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]])
        results.append((bbox, det.get("text", ""), det.get("confidence", 0.5)))

    return results


def synthesis_compare(prompt: str) -> str:
    """
    Text-only chat completion for audio-vs-video comparison.
    Returns raw text response.
    """
    print("[GeminiClient] Running synthesis comparison...")
    client = get_client()

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config={
            "temperature": 0.1,
            "max_output_tokens": 500,
        }
    )

    return response.text.strip()


def quantifier_analyze(image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Multimodal call with image + prompt for 5-axis visual quantifier.
    Returns parsed JSON with scores.
    """
    print(f"[GeminiClient] Running quantifier analysis: {image_path}")
    client = get_client()

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            {
                "parts": [
                    {
                        "text": f"{system_prompt}\n\n{user_prompt}"
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }
        ],
        config={
            "temperature": 0.2,
            "max_output_tokens": 600,
        }
    )

    content = response.text.strip()
    # Strip markdown code fences
    content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)
