import os
import json
import shutil
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import asyncio
import traceback

from processing.processor_video import process_video
from processing.processor_image import process_image
from processing import gemini_client

# Estado del servidor
server_status: Dict[str, Any] = {"status": "starting", "message": "Iniciando..."}

# Log global de síntesis
synthesis_log: List = []

# Chunked upload storage: {upload_id: {dir, filename, total_chunks, received}}
active_uploads: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate Gemini API key and test connectivity on startup."""
    print("Iniciando servidor... Validando Gemini API key...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        server_status["status"] = "error"
        server_status["message"] = "GEMINI_API_KEY no configurada. Set env var GEMINI_API_KEY."
        print(f"ERROR: {server_status['message']}")
    else:
        try:
            client = gemini_client.get_client()
            # Test connectivity with a minimal call
            response = client.models.generate_content(
                model=gemini_client.MODEL,
                contents=["Respond with OK"]
            )
            server_status["status"] = "ready"
            server_status["message"] = "Gemini 2.0 Flash conectado."
            print(f"\n--- Servidor Analyzer Cloud (Gemini 2.0 Flash) Listo! ---")
        except Exception as e:
            server_status["status"] = "error"
            server_status["message"] = f"Error conectando con Gemini API: {e}"
            print(f"ERROR: {server_status['message']}")

    yield

    print("Apagando servidor.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def get_status():
    if server_status["status"] == "error":
        return {"status": "error", "message": server_status["message"]}
    return {"status": "ready", "message": server_status["message"]}


@app.get("/api/synthesis-log")
async def get_synthesis_log():
    return {"log": synthesis_log}


@app.post("/api/analyze/images")
async def analyze_images(files: List[UploadFile] = File(...), settings: str = Form(...)):
    if server_status["status"] == "error":
        raise HTTPException(status_code=503, detail=f"Servidor no listo: {server_status['message']}")

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
                settings=image_settings
            )
            results.append(report)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    return {"results": results}


@app.post("/api/upload/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...)
):
    """Receive a single chunk of a large file upload."""
    # Initialize upload session
    if upload_id not in active_uploads:
        upload_dir = tempfile.mkdtemp()
        active_uploads[upload_id] = {
            "dir": upload_dir,
            "filename": filename,
            "total_chunks": total_chunks,
            "received": set()
        }

    session = active_uploads[upload_id]
    chunk_path = os.path.join(session["dir"], f"chunk_{chunk_index:04d}")

    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(chunk.file, f)

    session["received"].add(chunk_index)
    done = len(session["received"]) == session["total_chunks"]

    return {"status": "ok", "chunk_index": chunk_index, "complete": done}


@app.post("/api/upload/complete")
async def upload_complete(upload_id: str = Form(...)):
    """Reassemble chunks into a single file. Returns the temp file path token."""
    if upload_id not in active_uploads:
        raise HTTPException(status_code=404, detail="Upload session not found.")

    session = active_uploads[upload_id]
    if len(session["received"]) != session["total_chunks"]:
        missing = session["total_chunks"] - len(session["received"])
        raise HTTPException(status_code=400, detail=f"Faltan {missing} chunks.")

    # Reassemble
    ext = os.path.splitext(session["filename"])[1]
    final_path = os.path.join(session["dir"], f"complete{ext}")

    with open(final_path, "wb") as out:
        for i in range(session["total_chunks"]):
            chunk_path = os.path.join(session["dir"], f"chunk_{i:04d}")
            with open(chunk_path, "rb") as cp:
                shutil.copyfileobj(cp, out)
            os.remove(chunk_path)

    session["final_path"] = final_path
    return {"status": "ok", "upload_id": upload_id}


@app.post("/api/analyze/video")
async def analyze_video(
    settings: str = Form(...),
    file: Optional[UploadFile] = File(None),
    upload_id: Optional[str] = Form(None)
):
    if server_status["status"] == "error":
        raise HTTPException(status_code=503, detail=f"Servidor no listo: {server_status['message']}")

    try:
        video_settings = json.loads(settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON inválido.")

    # Determine video path: from chunked upload or direct file
    tmp_video_path = None
    cleanup_dir = None

    if upload_id and upload_id in active_uploads:
        session = active_uploads[upload_id]
        tmp_video_path = session.get("final_path")
        cleanup_dir = session.get("dir")
        if not tmp_video_path or not os.path.exists(tmp_video_path):
            raise HTTPException(status_code=400, detail="Upload incompleto. Llama a /api/upload/complete primero.")
    elif file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_video_path = tmp.name
    else:
        raise HTTPException(status_code=400, detail="Se requiere file o upload_id.")

    try:
        full_report = await asyncio.to_thread(
            process_video,
            video_path=tmp_video_path,
            settings=video_settings
        )
        if full_report.get("synthesis_log"):
            synthesis_log.extend(full_report["synthesis_log"])
        return full_report

    except Exception as e:
        print(f"Error procesando video: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if cleanup_dir and os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)
            active_uploads.pop(upload_id, None)
        elif tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


if __name__ == "__main__":
    import uvicorn
    print("Iniciando con uvicorn...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
