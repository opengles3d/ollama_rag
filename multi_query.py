from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import re


# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
embeddings_model = OllamaEmbeddings(model="qwen3:4b")

db = InMemoryVectorStore(embedding=embeddings_model)
db.add_documents(documents)
# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 5})

# instruction to generate multiple queries
perspectives_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based  similarity search. 
    Provide these alternative questions separated by newlines. 
    Original question: {question}""")

llm = ChatOpenAI(temperature=0, model="qwen3:4b", base_url="http://localhost:11434/v1", api_key="123")


def parse_queries_output(message):
    #print(f"Raw output from LLM: {message.content}")
    # 去除<think>...</think>之间的内容
    cleaned = re.sub(r"<think>.*?</think>", "", message.content, flags=re.DOTALL)
    # 按行分割，去除空行和首尾空白
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    return lines

query_gen = perspectives_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    # Flatten list of lists, and dedupe them
    deduped_docs = {
        doc.page_content: doc for sublist in document_lists for doc in sublist}
    # return a flat list of unique docs
    return list(deduped_docs.values())


retrieval_chain = query_gen | retriever.batch | get_unique_union


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the key figures in the ancient greek history of philosophy?"


@chain
def multi_query_qa(input):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
print("Running multi query qa\n")
result = multi_query_qa.invoke(query)
print(result.content)
