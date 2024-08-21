import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List
from env import OPENAI_API_KEY  # Recupera a key armazenada em um arquivo que está no .gitignore 

#######################################################################
# Extração do Texto 
#######################################################################
def extract_text(input_path):
    parsed_url = urlparse(input_path)
    if parsed_url.scheme in ['http', 'https']:
        # Se for uma URL, faz o download do conteúdo
        response = requests.get(input_path)
        response.raise_for_status()  # Levanta um erro se a requisição falhar

        # Extraí o texto da página web
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
    else:
        # Se não for uma URL, assume que é um caminho de arquivo PDF
        with open(input_path, 'rb') as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() or ''

    return text

#######################################################################
# Dividir o Texto em Partes Menores
#######################################################################
def split_text(text, max_tokens=500):
    # Vamos dividir o texto em parágrafos, assumindo que cada parágrafo seja pequeno o suficiente.
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ''
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_tokens:
            current_chunk += paragraph + '\n\n'
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph + '\n\n'
    
    # Adiciona o último chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

#######################################################################
# Indexação dos Chunks
#######################################################################
def create_index(chunks: List[str]) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Criar objetos Document
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Criar o índice FAISS
    faiss_index = FAISS.from_documents(documents, embeddings)
    
    return faiss_index

#######################################################################
# Recuperação dos Chunks
#######################################################################
def retrieve_relevant_chunks(query: str, index: FAISS) -> List[str]:
    results = index.similarity_search(query, k=3)
    
    relevant_chunks = [result.page_content for result in results]
    return relevant_chunks

#######################################################################
# Geração da Resposta Usando a API da OpenAI
#######################################################################
# Define a estrutura de resposta para a API da OpenAI
class AnswerResponse(BaseModel):
    """Respond in a conversational manner to answer the question based on the context provided."""
    response: str = Field(description="A response to the user's question based on the context provided. If the answer is not in the context, say 'Sei lá'")

# Setup do modelo da OpenAI usando LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Modelo GPT-4o-mini
    openai_api_key=OPENAI_API_KEY
)

# Configurando o modelo LLM para fornecer a resposta estruturada
structured_llm = llm.with_structured_output(AnswerResponse)

# Função para gerar a resposta usando o LangChain
def generate_answer(question: str, relevant_chunks: List[str]) -> str:
    # Combinar os chunks relevantes em um contexto
    context = "\n".join(relevant_chunks)
    
    # Preparar a entrada do prompt com o contexto e a pergunta
    input_data = f"Context: {context}\n\n Question: {question}\n"

    # Invocar o modelo para gerar a resposta
    answer = structured_llm.invoke(input_data)
    
    return answer.response

#######################################################################
# Integração Final
#######################################################################
def answer_question(pdf_path, question):
    # 1. Extrair texto do PDF
    text = extract_text(pdf_path)
    
    # 2. Dividir o texto em chunks
    chunks = split_text(text)

    # # 3. Criar o índice e embeddings
    index = create_index(chunks)
    
    # # 4. Recuperar os trechos relevantes
    relevant_chunks = retrieve_relevant_chunks(question, index)
    
    # # 5. Gerar a resposta
    answer = generate_answer(question, relevant_chunks)
    
    return answer

#######################################################################
# Usando o Programa
#######################################################################

# Teste 1
pdf_path = 'https://www.technologyreview.com/2016/08/09/158125/ais-language-problem/?gad_source=1&gclid=Cj0KCQjwzva1BhD3ARIsADQuPnU8G_YKoOW29JnFezUFmM8zVN36cX0gQkpzktMoJBlpgRkPr4oDlKMaAgUlEALw_wcB'
question = 'Quais são os principais desafios da IA na compreensão do contexto em linguagem humana?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 1: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'Como combinar aprendizado de máquina com regras linguísticas pode melhorar a compreensão da linguagem pela IA?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 2: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'Quais exemplos o artigo dá sobre falhas da IA em entender linguagem, e o que podemos aprender com eles?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 3: {question}\n")
print(f"Resposta: {resposta}\n")

# Teste 2
pdf_path = 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/'
question = 'Qual é a principal vantagem das Redes Neurais Recorrentes (RNNs) em comparação com redes neurais tradicionais para tarefas de processamento de sequências?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 1: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'Como as variantes das RNNs, como LSTM e GRU, melhoram o desempenho em tarefas que envolvem dependências de longo prazo?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 2: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'Quais são alguns exemplos de aplicações práticas onde as RNNs demonstraram resultados surpreendentes, conforme discutido no artigo?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 3: {question}\n")
print(f"Resposta: {resposta}\n")

# Teste 3
pdf_path = 'lista_5/teste_3.pdf'
question = 'Por que as funções de ativação não lineares, como a ReLU ou a sigmoide, são essenciais para o funcionamento de redes neurais?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 1: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'O que torna o problema XOR um exemplo importante na compreensão das limitações dos perceptrons e da necessidade de redes neurais multicamadas?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 2: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'Quais são as principais diferenças entre uma rede neural feedforward e uma rede neural recorrente (RNN)?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 3: {question}\n")
print(f"Resposta: {resposta}\n")

# Teste 4
pdf_path = 'lista_5/teste_3.pdf'
question = 'Do que o Silvio Santos Morreu?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 1: {question}\n")
print(f"Resposta: {resposta}\n")
question = 'Quem é o pai do Ash?'
resposta = answer_question(pdf_path, question)
print(f"Pergunta 2: {question}\n")
print(f"Resposta: {resposta}\n")

