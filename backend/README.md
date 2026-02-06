# Analyzer 2.0 - Servidor Backend

Este es el servidor backend para la aplicación Analyzer 2.0, construido con FastAPI y optimizado para Apple Silicon (M4 Max con Metal).

## 1. Requisitos Previos

- **Hardware:** Mac con Apple Silicon (M1/M2/M3/M4).
- **Software:**
    - Python 3.10+
    - Homebrew (para instalar dependencias)
    - **FFmpeg:** Utilidad esencial para el procesamiento de audio/video.
      ```bash
      brew install ffmpeg
      ```

## 2. Instalación

1.  **Clonar/Crear Directorio del Backend:**
    Crea tu estructura de carpetas `backend/`.

2.  **Crear un Entorno Virtual de Python:**
    ```bash
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar `llama-cpp-python` con Soporte Metal (¡CRUCIAL!)**
    Este es el paso más importante. Debemos compilarlo con soporte para la GPU de tu Mac (Metal).

    ```bash
    # Configura las flags para la compilación con Metal
    export CMAKE_ARGS="-DLLAMA_METAL=on"
    export FORCE_CMAKE=1
    
    # Instala el paquete
    pip install llama-cpp-python
    ```

4.  **Instalar `whisper-cpp-python`:**
    ```bash
    # Similar, compilamos con Metal
    export WHISPER_METAL=on
    
    # Instala el paquete
    pip install whisper-cpp-python
    ```

5.  **Instalar el Resto de Dependencias:**
    Crea el archivo `requirements.txt` (proporcionado abajo) en tu carpeta `backend/` y corre:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Descargar los Modelos (GGUF)

Tu M4 Max con 36GB de RAM puede manejar estos modelos. Descárgalos y colócalos dentro de una nueva carpeta `backend/models/`.

1.  **Whisper (Transcripción):** `large-v3-q5_K_M.gguf` (Aprox. 2.5GB)
    - [Descargar desde HuggingFace](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-q5_K_M.bin) (Renombra a `.gguf` si es necesario)

2.  **LLaVA-NeXT (Visión/OCR):** `LLaVA-NeXT-34B-Q4_K_M.gguf` (Aprox. 20GB)
    - [Descargar desde HuggingFace](https://huggingface.co/cjpais/llava-v1.6-34B-gguf/blob/main/llava-v1.6-34B-Q4_K_M.gguf)

3.  **Llama 3.1 (Síntesis):** `Llama-3.1-8B-Instruct-Q6_K.gguf` (Aprox. 6.2GB)
    - [Descargar desde HuggingFace](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct.Q6_K.gguf)

Tu carpeta de `models/` debería verse así: