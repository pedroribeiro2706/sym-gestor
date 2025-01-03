from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pymysql
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pydantic import BaseModel
# Importando e renomeando o Pinecone para gerenciamento do índice
from pinecone import Pinecone as PineconeClient, ServerlessSpec


load_dotenv()  # Carregar variáveis de ambiente

router = APIRouter()

# Conectar ao MySQL
def get_mysql_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

pc = PineconeClient(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1"  # Substitua pela região configurada no Pinecone
)

index_name = "sym-comentarios"

# Verifica se o índice existe
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Conecta ao índice
index = pc.Index(index_name)

# Configurando o vector_store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="comentario",
    namespace="comentarios_namespace"
)

# Criar modelo LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)

# Função para buscar dados agregados do MySQL
def fetch_sentiment_summary():
    connection = get_mysql_connection()
    with connection.cursor() as cursor:
        query = """
        SELECT unidade, sentimento, COUNT(*) AS total
        FROM comentarios_clientes
        GROUP BY unidade, sentimento;
        """
        cursor.execute(query)
        results = cursor.fetchall()
    connection.close()
    return results

# Função para buscar comentários similares no Pinecone
def fetch_pinecone_data(query_text):
    results = vector_store.similarity_search_with_score(query_text, k=3)

    # Log dos resultados para depuração
    print("Resultados do Pinecone:", results)

    # Validação para garantir que 'page_content' existe antes de acessar
    filtered_results = [
        res.page_content for res, score in results
        if hasattr(res, "page_content") and score >= 0.8
    ]

    return filtered_results


# Definir o modelo esperado pelo corpo JSON
class ReportInput(BaseModel):
    meta: str
    query: str

# Endpoint atualizado para receber os dados no corpo JSON
@router.post("/generate-report")
async def generate_report(input_data: ReportInput):
    try:
        meta = input_data.meta  # Extrai o valor da meta
        query = input_data.query  # Extrai o valor da consulta no Pinecone

        print("Meta recebida:", meta)
        print("Query recebida:", query)

        # Buscar dados do MySQL
        mysql_summary = fetch_sentiment_summary()
        print("Resumo do MySQL:", mysql_summary)

        # Buscar dados do Pinecone
        pinecone_comments = fetch_pinecone_data(query)
        print("Comentários do Pinecone:", pinecone_comments)

        # Criar prompt templates e chains
        prompt_metas = PromptTemplate(
            input_variables=["metas", "comentarios"],
            template=(
                "Metas da empresa: {metas}\n\n"
                "Resumo dos comentários por unidade: {comentarios}\n\n"
                "Analise se as metas foram atingidas com base nos comentários. Gere uma resposta formal."
            )
        )

        prompt_market = PromptTemplate(
            input_variables=["comentarios", "pinecone"],
            template=(
                "Analise os seguintes comentários por unidade: {comentarios}\n\n"
                "Compare com os seguintes exemplos do banco vetorial: {pinecone}\n\n"
                "Apresente tendências e insights do mercado."
            )
        )

        # Criar chains
        chain_metas = prompt_metas | llm
        chain_market = prompt_market | llm

        # Invocar respostas
        print("Executando chain metas...")
        resultado_metas = chain_metas.invoke({
            "metas": meta, 
            "comentarios": str(mysql_summary)
        })
        print("Resultado Metas:", resultado_metas)

        print("Executando chain mercado...")
        resultado_market = chain_market.invoke({
            "comentarios": str(mysql_summary), 
            "pinecone": str(pinecone_comments)
        })
        print("Resultado Mercado:", resultado_market)

        # Relatório final
        report = (
            "### Análise de Metas\n" + str(resultado_metas.content) +  # O atributo content é utilizado para extrair o texto da AIMessage e converte para string
            "\n\n### Análise de Mercado\n" + str(resultado_market.content)
        )


        return JSONResponse(content={"report": report})

    except Exception as e:
        print("Erro durante a execução:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
