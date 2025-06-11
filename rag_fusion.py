from langchain_community.document_loaders import TextLoader
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

prompt_rag_fusion = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query. \n Generate multiple search queries related to: {question} \n Output (4 queries):""")


def parse_queries_output(message):
    #return message.content.split('\n')
    # 去除<think>...</think>之间的内容
    cleaned = re.sub(r"<think>.*?</think>", "", message.content, flags=re.DOTALL)
    # 按行分割，去除空行和首尾空白
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    return lines


llm = ChatOpenAI(temperature=0, model="qwen3:4b", base_url="http://localhost:11434/v1", api_key="123")
query_gen = prompt_rag_fusion | llm | parse_queries_output

#query = "Who are the key figures in the ancient greek history of philosophy?"

#generated_queries = query_gen.invoke(query)

#print("generated queries: ", generated_queries)


"""
we fetch relevant documents for each query and pass them into a function to rerank (that is, reorder according to relevancy) the final list of relevant documents.
"""


def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents and an optional parameter k used in the RRF formula"""
    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
    # sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

#result = retrieval_chain.invoke(query)

#print("retrieved context using rank fusion: ", result[0].page_content)
#print("\n\n")

#print("Use model to answer question based on retrieved docs\n")


@chain
def rag_fusion(input):
    prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
   )
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
query = "Who are the some important yet not well known philosophers in the ancient greek history of philosophy?"
print("Running rag fusion\n")
result = rag_fusion.invoke(query)
print(result.content)
