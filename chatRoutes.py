from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configura o router
router = APIRouter()

# Inicializa o modelo GPT
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Função para responder com base no input
def chat_response(user_message: str) -> str:
    try:
        # Prepara a mensagem para o modelo
        messages = [
            HumanMessage(content="Você é um agente especialista em atendimento a restaurantes."),
            HumanMessage(content=f"Cliente: {user_message}")
        ]
        # Gera a resposta
        response = chat_model.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Erro no agente do chat: {str(e)}")
        return "Ocorreu um erro ao processar a mensagem. Tente novamente."

# Define o esquema de entrada do chat
class ChatInput(BaseModel):
    message: str

# Endpoint para o chat
@router.post("/chat-agent")
async def chat_agent(request: ChatInput):
    try:
        # Processa a mensagem do usuário
        user_message = request.message
        agent_reply = chat_response(user_message)

        # Retorna a resposta ao frontend
        return JSONResponse(content={"reply": agent_reply})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
