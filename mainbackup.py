from openai import OpenAI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa o cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

TEMP_DIR = "./temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(TEMP_DIR, file.filename)
        print(f"[INFO] Salvando arquivo em: {file_path}")

        # Salva o arquivo enviado
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print("[INFO] Arquivo salvo com sucesso!")

        # Transcrição do áudio
        print("[INFO] Iniciando transcrição com Whisper...")
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print("[INFO] Transcrição concluída!")

        return JSONResponse(
            content={
                "message": f"Arquivo recebido: {file.filename}",
                "transcription": transcription.text,
            }
        )
    except Exception as e:
        print(f"[ERROR] Erro durante o processo: {str(e)}")
        return JSONResponse(
            content={"error": f"Erro durante a transcrição: {str(e)}"},
            status_code=500,
        )
