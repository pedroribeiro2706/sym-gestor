import openai
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pymysql
import os
import uuid
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import StructuredTool
from pydantic import BaseModel
import pinecone
from datetime import datetime, timezone
from pinecone import Pinecone, ServerlessSpec
import warnings

from reportRoutes import router as report_router
from dashboardRoutes import router as dashboard_router
from chatRoutes import router as chat_router

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configura a chave da API do OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Instancia o cliente openai
client = openai.Client(api_key=openai.api_key)

# Configurar conexão com o banco MySQL
def get_mysql_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# Inicialize o cliente Pinecone
pinecone_client = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")  # Certifique-se de que a variável está no .env
)

# Verifique se o índice existe ou crie um novo
if "sym-comentarios" not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name="sym-comentarios",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Conecte-se ao índice
pinecone_index = pinecone_client.Index("sym-comentarios")


# Valide a existência do namespace no código
def validate_namespace(index, namespace):
    try:
        namespaces = index.describe_index_stats().get('namespaces', {})
        print(f"[DEBUG] Namespaces disponíveis no Pinecone: {namespaces}")
        if namespace not in namespaces:
            raise ValueError(f"Namespace '{namespace}' não encontrado no Pinecone.")
    except Exception as e:
        print("[ERROR] Erro ao validar namespace:", str(e))
        raise





############################################## CRIA A ESTRUTURA DO BACKEND ########################################


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clockdesign.com.br"],  # Permite apenas o frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers para o dashboard
app.include_router(report_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

# Defina o esquema para o argumento
class SentimentAnalysisInput(BaseModel):
    transcription: str

# Classe para validar os dados enviados pelo frontend
class UserDetails(BaseModel):
    record_id: int
    nome_cliente: str
    email: str
    unidade: str

# Cria a pasta temporária para os áudios
TEMP_DIR = "./temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)



################################## FUNÇÕES DE INSERÇÃO NO MySQL ###############################################


# Função para salvar dados iniciais no banco MySQL
def save_initial_to_mysql(connection, comentario):
    try:

        # Comando SQL para inserir os dados iniciais
        sql = """
        INSERT INTO comentarios_clientes (comentario)
        VALUES (%s)
        """
        # Dados para inserção
        data = (comentario,)

        # Usando o gerenciador de contexto para o cursor
        with connection.cursor() as cursor:
            # Executar a inserção
            cursor.execute(sql, data)
            connection.commit()
            # Obter o ID gerado automaticamente
            record_id = cursor.lastrowid
            print(f"[DEBUG] Tentando salvar no banco de dados (1) com ID: {record_id}")
            print(f"[INFO] Dados iniciais salvos com sucesso para Record ID: {record_id}")

        return record_id  # Retorna o ID gerado para uso posterior

    except Exception as e:
        print(f"Erro ao salvar os dados iniciais no banco: {str(e)}")
        return None  # Retorna None em caso de erro


# Função para atualizar o sentimento no banco MySQL
def update_sentiment_to_mysql(connection, record_id, sentimento):
    try:
        # Comando SQL para atualizar o sentimento
        sql = """
        UPDATE comentarios_clientes
        SET sentimento = %s
        WHERE id = %s
        """
        # Dados para atualização
        data = (sentimento, record_id)

        print(f"[DEBUG] SQL Query: {sql}")
        print(f"[DEBUG] Dados para atualização: {data}")

        # Usando o gerenciador de contexto para o cursor
        with connection.cursor() as cursor:
            # Executar a atualização
            cursor.execute(sql, data)
            connection.commit()
            print(f"[INFO] Sentimento atualizado com sucesso para Record ID: {record_id}")

    except Exception as e:
        print(f"Erro ao atualizar o sentimento no banco: {str(e)}")


# Função para criar o cadastro do cliente no mesmo ID que o comentário e o sentimento
def update_user_details_to_mysql(connection, record_id, nome_cliente, email, unidade):
    try:
        print(f"[DEBUG] Atualizando os detalhes para Record ID: {record_id}")

        # Declaração das variáveis SQL e dados antes do bloco 'with'
        sql = """
        UPDATE comentarios_clientes
        SET nome_cliente = %s, email = %s, unidade = %s
        WHERE id = %s
        """
        data = (nome_cliente, email, unidade, record_id)

        # Bloco 'with' para o cursor
        with connection.cursor() as cursor:
            affected_rows = cursor.execute(sql, data)  # Verifica as linhas afetadas
            cursor.execute(sql, data)
            connection.commit()
            print(f"[DEBUG] Tentando salvar dados do usuário no banco de dados com ID: {record_id}")
            if affected_rows == 0:
                print(f"[ERROR] Nenhuma linha foi atualizada para Record ID: {record_id}")
            else:
                print(f"[INFO] {affected_rows} linha(s) atualizada(s) no banco para Record ID: {record_id}")

    except Exception as e:
        print(f"Erro ao atualizar os detalhes do usuário no banco: {str(e)}")
        raise  # Relança a exceção para depuração mais detalhada


############################################ FUNÇÂO PARA INSERÇÃO NO PINECONE ######################################################


# Função para Gerar o Vetor do Comentário
def gerar_vetor_comentario(comentario: str) -> list:
    try:
        response = client.embeddings.create(
            input=comentario,
            model="text-embedding-ada-002"
        )
        vetor = response.data[0].embedding  # Retorna o vetor
        # print(f"[DEBUG] Vetor gerado com sucesso: {vetor[:5]}...")  # Mostra os primeiros valores
        return vetor
    except Exception as e:
        print(f"[ERROR] Erro ao gerar vetor: {str(e)}")
        return None


# Função para salvar o vetor e metadata do comentário no Pinecone
def save_comment_to_pinecone(record_id, comentario, vetor, sentimento):
    try:
        if vetor is None:
            raise ValueError("O vetor gerado é None. Não é possível salvar no Pinecone.")

        # Criação da metadata
        metadata = {
            "comentario": comentario,
            "sentimento": sentimento,
            "timestamp": str(datetime.now(timezone.utc)),  # Hora UTC para consistência
        }

        # Validação antes de salvar os dados
        validate_namespace(pinecone_index, "comentarios_namespace")
        print("[DEBUG] Namespace validado com sucesso.")

        print(f"[DEBUG] Metadata a ser salva: {metadata}")

        # Cria o vetor no Pinecone
        response = pinecone_index.upsert(
            vectors=[
                {
                    "id": str(record_id),
                    "values": vetor,
                    "metadata": metadata,
                }
            ],
            namespace="comentarios_namespace"  # Adiciona o namespace
        )
        print(f"[INFO] Comentário salvo no Pinecone com ID: {record_id}")
        print(f"[DEBUG] Resposta do Pinecone após upsert: {response}")

    except Exception as e:
        print(f"[ERROR] Erro ao salvar no Pinecone: {str(e)}")


# Função para salvar as informações do cliente (metadata)
def update_user_metadata_in_pinecone(record_id, nome_cliente, email, unidade):
    try:
        # Verifica se o namespace é válido
        validate_namespace(pinecone_index, "comentarios_namespace")

        # Verificar se o ID existe no Pinecone usando fetch
        print(f"[DEBUG] Verificando existência do ID no Pinecone: {record_id}")
        fetch_result = pinecone_index.fetch(ids=[str(record_id)], namespace="comentarios_namespace")
        if not fetch_result or str(record_id) not in fetch_result.get('vectors', {}):
            raise ValueError(f"O ID {record_id} não existe no namespace 'comentarios_namespace'")

        # stats = pinecone_index.describe_index_stats(namespace="comentarios_namespace")
        # Depuração para verificar as estatísticas do índice
        # print(f"[DEBUG] Estatísticas completas do índice Pinecone: {stats}")
        # Debug para verificar o namespace e o ID
        # print(f"[DEBUG] Verificando existência do ID no namespace 'comentarios_namespace': {record_id}")
        # if str(record_id) not in stats.get('namespaces', {}).get('comentarios_namespace', {}).get('ids', []):
        #     raise ValueError(f"O ID {record_id} não existe no namespace 'comentarios_namespace'")

        # Atualiza apenas a metadata do vetor
        response = pinecone_index.update(
            id=str(record_id),
            set_metadata={
                "nome": nome_cliente,
                "email": email,
                "unidade": unidade,
            },
            namespace="comentarios_namespace"  # Adiciona o namespace
        )
        print(f"[INFO] Metadata atualizada no Pinecone para Record ID: {record_id}")
        print(f"[DEBUG] Resposta do Pinecone ao atualizar metadata: {response}")
    except Exception as e:
        print(f"[ERROR] Erro ao atualizar metadata no Pinecone: {str(e)}")



############################################ FUNÇÃO PARA ANÁLISE DE SENTIMENTO #####################################################


# Define a função de análise de sentimento
def analyze_sentiment_function(transcription: str) -> str:
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Leia e classifique o seguinte texto: {text} como: 'positivo', 'negativo' ou 'neutro'. Responda em português"
        )   
    )
    prompt_text = prompt.format(text=transcription)
    # response = chat_model([HumanMessage(content=prompt_text)])
    response = chat_model.invoke([HumanMessage(content=prompt_text)])
    return response.content.strip().lower()


########################################## CONSTRUÇÃO DO AGENTE PARA ANÁLISE DE SENTIMENTO ###########################################


# Inicializa o modelo LLM OpenAI
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Registra a ferramenta de análise de sentimento no LangChain
tools = [
    StructuredTool(
        name="SentimentAnalyzer",
        func=analyze_sentiment_function,
        description="Classifica como 'positivo', 'negativo' ou 'neutro' o sentimento de um texto.",
        args_schema=SentimentAnalysisInput  # Usando o modelo correto
    )
]

# Inicializa o agente
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent="chat-zero-shot-react-description",
    # agent="structured-chat-zero-shot-react-description",
    # agent="zero-shot-react-description",
    # agent="zero-shot-prompt",
    verbose=True,
    handle_parsing_errors=True
)



############################################ ENDPOINTS PARA O FRONTEND ###########################################################

# Rota raiz para teste
@app.get("/")
async def root():
    return {"message": "Backend SYM-Gestor funcionando corretamente!"}

#Endpoint para upload e transcrição do áudio
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Caminho do arquivo
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
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        transcription = response.text  # Acessa o texto diretamente
        print("[INFO] Transcrição concluída!")

        # Salva a transcrição como arquivo de texto
        transcription_file_path = os.path.splitext(file_path)[0] + ".txt"
        with open(transcription_file_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"[INFO] Transcrição salva em: {transcription_file_path}")

        return JSONResponse(
            content={
                "message": f"Arquivo recebido: {file.filename}",
                "transcription": transcription,
            }
        )
    except Exception as e:
        print(f"[ERROR] Erro durante o processo: {str(e)}")
        return JSONResponse(
            content={"error": f"Erro durante a transcrição: {str(e)}"},
            status_code=500,
        )


#Endpoint para o agente para análise de sentimento receber a trasncrição, analisar e gravar no banco de dados MySQL
@app.post("/analyze-sentiment/")
async def analyze_sentiment(transcription: str = Query(...)):
    try:
        # Salvar o comentário no banco e obter o ID gerado automaticamente
        record_id = save_initial_to_mysql(get_mysql_connection(), transcription)
        if not record_id:
            raise ValueError("Erro ao salvar o comentário inicial no banco de dados.")
        print(f"[INFO] ID gerado pelo banco de dados: {record_id}")

        # Salvar o comentário inicial no banco de dados
        # save_initial_to_mysql(get_mysql_connection(), transcription )

        # Usa o agente para analisar o sentimento
        print("[INFO] Enviando transcrição para análise de sentimento...")
        print(f"[DEBUG] Transcrição recebida pelo agente: {transcription}")
        result = agent.invoke({"input": transcription})
        print(f"[DEBUG] Resultado retornado pelo agente: {result}")

        # Certifique-se de que o resultado é uma string simples
        if isinstance(result, str):
            output = result.lower().strip()
        elif isinstance(result, dict) and "output" in result:
            output = result["output"].lower().strip()
        else:
            raise ValueError("Formato inesperado no resultado do agente.")
        
        # Remover pontuação final do output, se existir
        output = output.rstrip(".,!?")

        # Validação para garantir que o sentimento seja uma das opções esperadas
        if output in ["positivo", "negativo", "neutro", "positive", "positive."]:
            sentiment = output
        else:
            sentiment = "indefinido"  # Caso o modelo não responda corretamente

        print(f"[DEBUG] Sentimento retornado pelo agente: {sentiment}")

        # Atualizar o sentimento no banco de dados
        update_sentiment_to_mysql(get_mysql_connection(), record_id, sentiment)

        # Gerar vetor do comentário
        vetor = gerar_vetor_comentario(transcription)  # Supondo que há uma função para gerar o vetor
        print(f"[DEBUG] Vetor gerado para o comentário: {vetor[:5]}...")

        # Salvar no Pinecone
        save_comment_to_pinecone(record_id, transcription, vetor, sentiment)
        print(f"[DEBUG] Dados a serem salvos no Pinecone:")
        print(f"  ID: {record_id}")
        print(f"  Valores do vetor: {vetor[:5]}...")


        return JSONResponse(
            content={
                "message": "Análise concluída e sentimento gravado.",
                "sentiment": sentiment,
                "record_id": record_id  # Inclui o record_id no retorno
            }
        )
    except Exception as e:
        print(f"[ERROR] Erro durante a análise de sentimento: {str(e)}")
        print(f"[ERROR] Erro ao gerar vetor: {str(e)}")
        return JSONResponse(
            content={"error": f"Erro durante a análise de sentimento: {str(e)}"},
            # vetor = None,  # Defina como None em caso de falha
            status_code=500,
        )


# Endpoint para atualizar os dados do usuário no banco de dados MySQL
@app.post("/update-user-details/")
async def update_user_details(data: UserDetails):
    print(data)
    try:
        print("Dados recebidos pelo backend:", data.model_dump())  # Substitui `data.dict()`
        # Conexão com o banco de dados
        connection = get_mysql_connection()

        # Extração dos valores enviados
        record_id = data.record_id
        nome_cliente = data.nome_cliente
        email = data.email
        unidade = data.unidade

        # Debug adicional
        print(f"Record ID: {record_id}")
        print(f"Nome Cliente: {nome_cliente}")
        print(f"Email: {email}")
        print(f"Unidade: {unidade}")

        print(f"[DEBUG] Record ID recebido no backend: {record_id}")
        # Atualizar os dados no banco
        update_user_details_to_mysql(connection, record_id, nome_cliente, email, unidade)
        print("[DEBUG] Atualização no MySQL concluída.")

        # Atualizar metadata no Pinecone
        update_user_metadata_in_pinecone(record_id, nome_cliente, email, unidade)
        print("[DEBUG] A função para atualização no Pinecone foi executada.")

        return JSONResponse(content={"message": "Dados atualizados com sucesso."})
    except Exception as e:
        print(f"[ERROR] Erro ao atualizar os detalhes do usuário: {str(e)}")
        return JSONResponse(
            content={"error": f"Erro ao atualizar os detalhes do usuário: {str(e)}"},
            status_code=500,
        )