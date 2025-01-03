from fastapi import APIRouter
import pymysql
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from datetime import date

# Carrega variáveis de ambiente
load_dotenv()

# Configura conexão MySQL
def get_mysql_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# Criação do router
router = APIRouter()

# Endpoint para buscar número de comentários por unidade
@router.get("/dashboard/comments-by-unit")
async def comments_by_unit():
    try:
        connection = get_mysql_connection()
        with connection.cursor() as cursor:
            query = """
                SELECT unidade, COUNT(*) AS total
                FROM comentarios_clientes
                GROUP BY unidade;
            """
            cursor.execute(query)
            results = cursor.fetchall()
        connection.close()
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint para buscar distribuição de sentimentos por unidade
@router.get("/dashboard/sentiment-by-unit")
async def sentiment_by_unit():
    try:
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
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint para buscar variação temporal de sentimentos
@router.get("/dashboard/sentiment-trend")
async def sentiment_trend():
    try:
        connection = get_mysql_connection()
        with connection.cursor() as cursor:
            query = """
                SELECT DATE(data_hora) AS data, sentimento, COUNT(*) AS total
                FROM comentarios_clientes
                GROUP BY DATE(data_hora), sentimento
                ORDER BY DATE(data_hora);
            """
            cursor.execute(query)
            results = cursor.fetchall()

            # Converte 'data' para string no formato ISO 8601 (YYYY-MM-DD)
            for row in results:
                if isinstance(row['data'], date):  # Verifica se é do tipo 'date'
                    row['data'] = row['data'].isoformat()  # Converte para string ISO

        connection.close()
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
