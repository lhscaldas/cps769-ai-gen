import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
import numpy as np
from typing import List, Dict
# Recupera a key armazenada em um arquivo que está no .gitignore 
from env import OPENAI_API_KEY  

#######################################################################
# Extração do PDF 
#######################################################################
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
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
# Recuperação de Documentos
#######################################################################
def retrieve_relevant_chunks(query: str, index: FAISS, chunks: List[str]) -> List[str]:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    query_embedding = embeddings.embed_query(query)
    results = index.similarity_search(query_embedding, k=3)
    
    relevant_chunks = [result['text'] for result in results]
    return relevant_chunks

#######################################################################
# Geração da Resposta Usando a API da OpenAI
#######################################################################

# Define a estrutura de resposta para a API da OpenAI
class AnswerResponse(BaseModel):
    """Respond in a conversational manner to answer the question based on the context provided."""
    response: str = Field(description="A response to the user's question based on the context provided")

# Setup do modelo da OpenAI usando LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Modelo GPT-4o-mini
    openai_api_key=OPENAI_API_KEY
)

# Configurando o modelo LLM para fornecer a resposta estruturada
structured_llm = llm.with_structured_output(AnswerResponse)

# Função para gerar a resposta usando o LangChain
def generate_answer(question, relevant_chunks):
    # Combinar os chunks relevantes em um contexto
    context = "\n".join(relevant_chunks)
    
    # Preparar a entrada do prompt com o contexto e a pergunta
    input_data = {
        "context": context,
        "question": question
    }
    
    # Invocar o modelo para gerar a resposta
    answer = structured_llm.invoke(input_data)
    
    return answer.response

#######################################################################
# Integração Final
#######################################################################
def answer_question_from_pdf(pdf_path, question):
    # 1. Extrair texto do PDF
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Dividir o texto em chunks
    chunks = split_text(text)
    
    # 3. Criar o índice e embeddings
    index = create_index(chunks)
    
    # 4. Recuperar os trechos relevantes
    relevant_chunks = retrieve_relevant_chunks(question, index, chunks)
    
    # 5. Gerar a resposta
    answer = generate_answer(question, relevant_chunks)
    
    return answer

#######################################################################
# Usando o Programa
#######################################################################
pdf_path = 'lista_5/teste_1.pdf'
question = 'Qual é o principal argumento do artigo?'

resposta = answer_question_from_pdf(pdf_path, question)
print(resposta)