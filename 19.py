from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="qwen3:4b"
    )

vector_store = InMemoryVectorStore(embeddings)

from langchain_core.documents import Document

document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
document_3 = Document(id="3", page_content="i will be deleted :(")

documents = [document_1, document_2, document_3]
vector_store.add_documents(documents=documents)

top_n = 10
for index, (id, doc) in enumerate(vector_store.store.items()):
    if index < top_n:
        # docs have keys 'id', 'vector', 'text', 'metadata'
        print(f"{id}: {doc['text']}")
    else:
        break


results = vector_store.similarity_search(query="thud",k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")


def _filter_function(doc: Document) -> bool:
    return doc.metadata.get("bar") == "baz"

results = vector_store.similarity_search(
    query="thud", k=1, filter=_filter_function
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")


results = vector_store.similarity_search_with_score(
    query="qux", k=1
)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")    
