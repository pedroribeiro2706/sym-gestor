import openai

# Configure sua chave de API
openai.api_key = "sua_chave_de_api"

# Caminho para o arquivo de áudio
file_path = "Elogio-Educado.m4a"

# Realiza a transcrição usando o método atualizado
with open(file_path, "rb") as audio_file:
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file
    )
    transcription = response.get("text", "")
    print(transcription)
