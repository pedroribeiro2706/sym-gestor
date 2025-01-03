import os
from openai import OpenAI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa o cliente OpenAI com a chave de API
api_key = os.getenv("OPENAI_API_KEY")

try:
    client = OpenAI(api_key=api_key)
    
    # Solicite uma conclusão de texto
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": "Flamengo foi campeão do mundo?"}
        ]
    )
    
    # Acesse o conteúdo da resposta corretamente
    for choice in response.choices:
        print(choice.message.content.strip())
except Exception as e:
    print(f"Erro de conexão: {e}")
