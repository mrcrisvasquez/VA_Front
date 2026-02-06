import os
import json
import torch
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
try:
    from llama_cpp.llama_chat_format import Llava15ChatHandler, Qwen2VLChatHandler
except ImportError:
    Llava15ChatHandler = None
    Qwen2VLChatHandler = None
    try:
         from llama_cpp.llama_chat_format import Llava15ChatHandler
    except ImportError:
         pass
import whisper
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import asyncio

# --- Importar EasyOCR ---
try:
    import easyocr
except ImportError:
    print("ERROR: La librería 'easyocr' no está instalada.")
    print("Instálala con: pip install easyocr")
    exit()

# Importar nuestros módulos de procesamiento
import sys
import traceback

try:
    from processing.processor_video import process_video
    from processing.processor_image import process_image
except ImportError as e:
    print("\n--- ERROR DE IMPORTACIÓN ---")
    print(f"No es que falte el archivo, es que falló una dependencia interna.")
    print(f"Detalle del error: {e}")
    print("Traceback completo:")
    traceback.print_exc() # Esto te dirá exactamente en qué línea explotó
    print("----------------------------\n")
    exit()

# --- Configuración de Modelos ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Nombres de los archivos de modelo
WHISPER_MODEL_NAME = "whisper-large-v3-q8_0.gguf"
QWEN_VL_MODEL_NAME = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"

# Rutas completas
WHISPER_PATH = os.path.join(MODEL_DIR, WHISPER_MODEL_NAME)
QWEN_VL_PATH = os.path.join(MODEL_DIR, QWEN_VL_MODEL_NAME)

# Variable global para almacenar los modelos cargados
models: Dict[str, Any] = {}

# Log global de síntesis
synthesis_log: List = []

def check_model_files() -> List[str]:
    """Verifica si los archivos de modelo basados en archivo existen.
    EasyOCR y Silero VAD se auto-descargan, solo validamos file-based models."""
    missing_files = []
    if not os.path.exists(WHISPER_PATH):
        missing_files.append(WHISPER_MODEL_NAME)
    if not os.path.exists(QWEN_VL_PATH):
        missing_files.append(QWEN_VL_MODEL_NAME)
    return missing_files

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga de modelos en memoria al iniciar la app.
    Cada modelo se carga independientemente para permitir operación parcial.
    """
    print("Iniciando servidor... Cargando modelos...")

    model_status = {}

    missing = check_model_files()
    if missing:
        warn_msg = f"Faltan archivos de modelo en /models: {', '.join(missing)}"
        print(f"ADVERTENCIA: {warn_msg}")
        for m in missing:
            if WHISPER_MODEL_NAME in m:
                model_status["whisper"] = {"status": "failed", "message": f"Archivo no encontrado: {m}"}
            if QWEN_VL_MODEL_NAME in m:
                model_status["qwen_vl"] = {"status": "failed", "message": f"Archivo no encontrado: {m}"}

    # 1. Cargar Whisper (Audio)
    if "whisper" not in model_status:
        try:
            print("Cargando OpenAI Whisper...")
            models["whisper"] = whisper.load_model("large", device="mps")
            model_status["whisper"] = {"status": "loaded", "message": "Whisper large cargado (MPS)"}
            print("Whisper cargado.")
        except Exception as e:
            print(f"ERROR cargando Whisper: {e}")
            model_status["whisper"] = {"status": "failed", "message": str(e)}

    # 2. Cargar EasyOCR (Visión - Texto)
    try:
        print("Cargando EasyOCR (ES + EN)...")
        models["ocr"] = easyocr.Reader(['es', 'en'], gpu=True)
        model_status["easyocr"] = {"status": "loaded", "message": "EasyOCR cargado (ES + EN, GPU)"}
        print("EasyOCR cargado.")
    except Exception as e:
        print(f"ERROR cargando EasyOCR: {e}")
        model_status["easyocr"] = {"status": "failed", "message": str(e)}

    # 3. Cargar Qwen-VL (Cerebro: Síntesis y Visión Compleja)
    if "qwen_vl" not in model_status:
        try:
            print(f"Cargando Qwen-VL: {QWEN_VL_MODEL_NAME}...")
            chat_handler = None
            if Qwen2VLChatHandler:
                try:
                    chat_handler = Qwen2VLChatHandler(clip_model_path=QWEN_VL_PATH)
                    print("Qwen2VLChatHandler inicializado para Qwen-VL.")
                except Exception as e:
                    print(f"ADVERTENCIA: Falló Qwen2VLChatHandler: {e}")
            elif Llava15ChatHandler:
                pass

            models["qwen_vl"] = Llama(
                model_path=QWEN_VL_PATH,
                chat_handler=chat_handler,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False
            )
            model_status["qwen_vl"] = {"status": "loaded", "message": "Qwen-VL cargado (4096 ctx)"}
            print("Qwen-VL cargado.")
        except Exception as e:
            print(f"ERROR cargando Qwen-VL: {e}")
            model_status["qwen_vl"] = {"status": "failed", "message": str(e)}

    # 4. Cargar Silero VAD (Detección de voz)
    try:
        print("Cargando Silero VAD...")
        models["silero_vad"], _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        model_status["silero_vad"] = {"status": "loaded", "message": "Silero VAD cargado (reservado para chunking futuro)"}
        print("Silero VAD cargado.")
    except Exception as e:
        print(f"ERROR cargando Silero VAD: {e}")
        model_status["silero_vad"] = {"status": "failed", "message": str(e)}

    # Determinar estado global
    models["_model_status"] = model_status
    loaded_count = sum(1 for v in model_status.values() if v["status"] == "loaded")
    total_count = len(model_status)

    if loaded_count == total_count:
        models["status_error"] = None
        print("\n--- ¡Servidor Analyzer 2.1 (EasyOCR + Qwen) Listo! ---")
    elif loaded_count > 0:
        failed = [k for k, v in model_status.items() if v["status"] == "failed"]
        models["status_warning"] = f"Modelos parcialmente cargados ({loaded_count}/{total_count}). Fallidos: {', '.join(failed)}"
        models["status_error"] = None
        print(f"\n--- Servidor con advertencia: {models['status_warning']} ---")
    else:
        models["status_error"] = "Ningún modelo pudo cargarse."
        print("\n--- ERROR: Ningún modelo cargado ---")

    yield

    print("Apagando servidor... Limpiando modelos.")
    models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
async def get_status():
    model_status = models.get("_model_status", {})
    if models.get("status_error"):
        return {"status": "error", "message": models.get("status_error"), "models": model_status}
    if models.get("status_warning"):
        return {"status": "warning", "message": models.get("status_warning"), "models": model_status}
    return {"status": "ready", "message": "Servidor listo (EasyOCR + Qwen-VL).", "models": model_status}

@app.get("/api/synthesis-log")
async def get_synthesis_log():
    return {"log": synthesis_log}


@app.post("/api/analyze/images")
async def analyze_images(files: List[UploadFile] = File(...), settings: str = Form(...)):
    if models.get("status_error"):
        raise HTTPException(status_code=503, detail=f"Servidor no listo: {models['status_error']}")
    
    try:
        image_settings = json.loads(settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON inválido en settings.")

    results = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            report = await asyncio.to_thread(
                process_image,
                image_path=tmp_path,
                settings=image_settings,
                ocr_model=models["ocr"],
                qwen_vl_model=models.get("qwen_vl")
            )
            results.append(report)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    return {"results": results}


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...), settings: str = Form(...)):
    if models.get("status_error"):
        raise HTTPException(status_code=503, detail=f"Servidor no listo: {models['status_error']}")

    try:
        video_settings = json.loads(settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON inválido.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_video_path = tmp.name
        
    try:
        full_report = await asyncio.to_thread(
            process_video,
            video_path=tmp_video_path,
            settings=video_settings,
            models=models
        )
        # Append synthesis log entries to global log
        if full_report.get("synthesis_log"):
            synthesis_log.extend(full_report["synthesis_log"])
        return full_report

    except Exception as e:
        print(f"Error procesando video: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(tmp_video_path): os.remove(tmp_video_path)
        raise HTTPException(status_code=500, detail=str(e))

    if os.path.exists(tmp_video_path): os.remove(tmp_video_path)

if __name__ == "__main__":
    import uvicorn
    print("Iniciando con uvicorn...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)