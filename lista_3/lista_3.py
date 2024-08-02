from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from env import OPENAI_API_KEY 
import pandas as pd
import pandasql as psql

# Inicialização do modelo de linguagem
llm = ChatOpenAI(model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
)

class WeatherQuery(BaseModel):
    """Estrutura para consultas meteorológicas."""
    csv_file: str = Field(description="O nome do arquivo CSV a ser lido")
    query: str = Field(description="A query SQL para consultar os dados no arquivo CSV")

class NaturalLanguageResponse(BaseModel):
    """Estrutura para a resposta em linguagem natural."""
    resposta: str = Field(description="A resposta em linguagem natural baseada na pergunta e no resultado")

# Exemplo de sistema para orientar a LLM
system = """You are an expert in meteorological data analysis. You understand the format of the CSV files: weather_YYYY, weather_sum_YYYY, weather_sum_all.

Here are some examples of queries and their structured responses:

example_user: Qual foi o dia com a maior temperatura em 2002?
example_assistant: {{"csv_file": "weather_2002.csv", "query": "SELECT DATA, MAX(TEMP) as Max_Temperature FROM df GROUP BY DATA ORDER BY Max_Temperature DESC LIMIT 1"}}

example_user: Qual foi o dia com a temperatura mais alta em 2020?
example_assistant: {{"csv_file": "weather_2020.csv", "query": "SELECT DATA, MAX(TEMP) as Max_Temperature FROM df GROUP BY DATA ORDER BY Max_Temperature DESC LIMIT 1"}}

example_user: Qual foi a média das temperaturas no ano de 2020?
example_assistant: {{"csv_file": "weather_sum_2020.csv", "query": "SELECT AVG(TEMP) as Avg_Temperature FROM df"}}

example_user: Qual foi a média da temperatura em janeiro de 2011?
example_assistant: {{"csv_file": "weather_2011.csv", "query": "SELECT AVG(TEMP) as Avg_Temperature FROM df WHERE strftime('%m', DATA) = '01'"}}

example_user: O mês de janeiro de 2020 foi frio?
example_assistant: {{"csv_file": "weather_2020.csv", "query": "SELECT MIN(TEMP) as Min_Temperature FROM df WHERE strftime('%m', DATA) = '01'"}}

example_user: Como se comparam as médias de temperatura de 2000 e 2019?
example_assistant: {{"csv_file": "weather_sum_all.csv", "query": "SELECT strftime('%Y', DATA) as Ano, AVG(TEMP) as Avg_Temperature FROM df WHERE strftime('%Y', DATA) IN ('2000', '2019') GROUP BY Ano"}}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

few_shot_structured_llm = prompt | llm.with_structured_output(WeatherQuery)

def process_weather_query(pergunta):
    response = few_shot_structured_llm.invoke(pergunta)
    
    # Determinar quais colunas ler com base no tipo de arquivo
    if 'sum' in response.csv_file:
        colunas = ['DATA (YYYY-MM-DD)', 'temp_avg']
        colunas_novas = ['DATA', 'TEMP']
    else:
        colunas = ['DATA (YYYY-MM-DD)', 'Hora UTC', 'TEMPERATURA DO PONTO DE ORVALHO (°C)']
        colunas_novas = ['DATA', 'HORA', 'TEMP']
    
    # Ler o arquivo CSV limitando às colunas especificadas e renomeá-las
    df = pd.read_csv(f"lista_3/archive/{response.csv_file}", usecols=colunas)
    df.columns = colunas_novas
    
    # Executar a query SQL usando pandasql
    result = psql.sqldf(response.query, {'df': df})
    return result, response

# Nova função para gerar a resposta em linguagem natural
def generate_natural_language_response(pergunta, result):
    # Converte o resultado da consulta em um formato de string compreensível
    result_str = result.to_string(index=False)
    
    # Prompt para gerar a resposta em linguagem natural
    nl_system = """You are a helpful assistant who provides responses in natural language based on data analysis results. Based on the user's question and the result of the data analysis, generate a concise and clear response in natural language.

Here are some examples:

example_user: Qual o dia mais quente do ano de 2001?
example_assistant: O dia mais quente do ano de 2001 foi [data] com uma temperatura de [temperatura] graus Celsius.

example_user: Qual a média da temperatura no ano de 2021?
example_assistant: A média da temperatura no ano de 2021 foi de [temperatura média] graus Celsius.

example_user: Compare a média de temperatura do ano de 2001 com a de 2021?
example_assistant: A média de temperatura no ano de 2001 foi de [temperatura média 2001] graus Celsius, enquanto em 2021 foi de [temperatura média 2021] graus Celsius."""

    nl_prompt = ChatPromptTemplate.from_messages([("system", nl_system), ("human", "{input}")])
    nl_structured_llm = nl_prompt | llm.with_structured_output(NaturalLanguageResponse)
    
    response = nl_structured_llm.invoke({"input": f"Pergunta: {pergunta}\nResultado: {result_str}"})
    return response.resposta

# Exemplos de uso
perguntas = [
    "Qual o dia mais quente do ano de 2001?",
    "Qual o dia mais quente do ano de 2021?",
    "Qual a média da temperatura no ano de 2021?",
    "Qual a média do mês de janeiro de 2010?",
    "Janeiro de 2021 fez frio?",
    "Compare a média de temperatura do ano de 2001 com a de 2021?"
]

for pergunta in perguntas:
    print("---------------------")
    print("Pergunta: ", pergunta)
    result, response = process_weather_query(pergunta)
    print("Query SQL: ", response.query)
    resposta = generate_natural_language_response(pergunta, result)
    print("Resposta: ",resposta)