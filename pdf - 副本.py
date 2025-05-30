import pdfplumber
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
import pandas as pd
from openai import OpenAI
from redis import Redis
from redis.commands.search.field import VectorField, TextField
import numpy as np
import os

client = OpenAI(
    api_key="123",
    base_url="http://localhost:11434/v1", 
)

r = Redis()
INDEX_NAME = "PDFData_VLforTable"
VECTOR_DIM = 1024
DISTANCE_METRIC = "COSINE"

pdf_path = "ICBC_2024_FYR.pdf"

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def detect_table_pages(pdf_path):
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if page.extract_tables():
                table_pages.append(i)
                break
    return table_pages

def save_page_as_image(pdf_path, page_number, output_path):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    pix.save(output_path)
    pdf_document.close()

def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)
    table_pages = set(detect_table_pages(pdf_path))
    for page_num in range(len(pdf_document)):
        if page_num in table_pages:
            continue
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
            break
    pdf_document.close()
    return images

text = extract_text_from_pdf(pdf_path)
table_pages = detect_table_pages(pdf_path)
images = extract_images_from_pdf(pdf_path)

print(f"Extracted text length: {len(text)}")
print(f"Detected {len(table_pages)} pages wiuth tables.")
print(f"Extracted {len(images)} images from the PDF.")

import base64
def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)

text_chunks = split_text(text)
print("First 20 text chunks:")
for chunk in text_chunks[:20]:
    print(chunk[:500] + "...")  # Print first 500 characters of each chunk

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def extract_table_from_page_image(page_num):
    image_path = f"table_page_{page_num}.png"
    save_page_as_image(pdf_path, page_num, image_path)
    base64_image = encode_image(image_path)

    try:
        #response = client.chat.completions.create(
        #    model="ZimaBlueAI/Qwen2.5-VL-7B-Instruct:latest",
        #    messages=[
        #        {
        #            "role": "user",
        #            "content": [
        #                {
        #                    "type": "text",
        #                    "text": "Extract the table from the image and return it in Markdown format. If no table is preset, retuen an empty string. Do not include additional explainations."
        #                },
        #                {
        #                    "type": "image_url",
        #                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        #                }
        #            ]
        #        }
        #    ],
        #    temperature=0.2,
        #)

        llm = ChatOllama(
            base_url="http://localhost:11434",
            model="qwen2.5vl:7b",
            temperature=0.2,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the table from the image and return it in Markdown format. If no table is preset, retuen an empty string. Do not include additional explainations."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]

        response = llm.invoke(messages)
        table_markdown = response.content
        #table_markdown = response.choices[0].message.content
        os.remove(image_path)  # Remove the image file after processing
        return table_markdown if table_markdown.strip() else ""
    except Exception as e:
        print(f"Error processing table on page {page_num}: {e}")
        os.remove(image_path)
        return ""
    
table_chunks = []
for page_num in table_pages[:1]:
    table_markdown = extract_table_from_page_image(page_num)
    if table_markdown:
        table_chunks.append(table_markdown)
        print(f"Extracted table from page {page_num}:{table_markdown[:500]}...")  # Print first 500 characters of the table

image_descriptions = []
for i in range(len(images[:1])):
    image_path = f"image_{i}.png"
    with open(image_path, "wb") as img_file:
        img_file.write(images[i])
        img_file.close()
    base64_image = encode_image(image_path)
    try:
        #response = client.chat.completions.create(
        #    model="ZimaBlueAI/Qwen2.5-VL-7B-Instruct:latest",
        #    messages=[
        #        {
        #            "role": "user",
        #            "content": [
        #                {
        #                    "type": "text",
        #                    "text": "提取图片中的信息，需要精准描述，不要漏掉信息，但是也不需要额外解释。若图片为饼状图、折现图和柱状图等，请使用饼状图、折现图和柱状图等关键词，并以 json 格式返回。若图片为表格，请使用表格的关键词，并以 markdown 格式返回。若图片为架构图、流程图等，请使用架构图、流程图等关键词，并以 mermaid 格式返回。"
        #                },
        #                {
        #                    "type": "image_url",
        #                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        #                }
        #            ]
        #        }
        #    ],
        #    temperature=0.2,
        #)

        #description = response.choices[0].message.content
        llm = ChatOllama(
            base_url="http://localhost:11434",
            model="qwen2.5vl:7b",
            temperature=0.2,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "提取图片中的信息，需要精准描述，不要漏掉信息，但是也不需要额外解释。若图片为饼状图、折现图和柱状图等，请使用饼状图、折现图和柱状图等关键词，并以 json 格式返回。若图片为表格，请使用表格的关键词，并以 markdown 格式返回。若图片为架构图、流程图等，请使用架构图、流程图等关键词，并以 mermaid 格式返回。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]

        response = llm.invoke(messages)
        description = response.content
        image_descriptions.append(description)
        print(f"Image {i} description: {description}")  # Print first 500 characters of the description
    except Exception as e:
        print(f"Error processing image {i}: {e}")
        image_descriptions.append("")
    
print(f"Split text into {len(text_chunks)} chunks.")
print(f"Converted {len(table_chunks)} tables to Markdown format.")
print(f"Extracted {len(image_descriptions)} image descriptions.")

# Cell 4: Vectorization (with progress bars)
def get_embedding(text, model="nomic-embed-text:latest"):
    if not text.strip():
        return None
    response = client.embeddings.create(
        input=text[:8192*2],  # Truncate to approximate token limit
        model=model,
        dimensions=VECTOR_DIM
    )
    return response.data[0].embedding

print("Generating text embeddings...")
text_embeddings = [get_embedding(chunk) for chunk in text_chunks]
print("Generating table embeddings...")
table_embeddings = [get_embedding(chunk) for chunk in table_chunks]
print("Generating image embeddings...")
image_embeddings = [get_embedding(description) for description in image_descriptions]

md_embedding_field = VectorField(
    "md_embedding",
    "FLAT",
    {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC
    },
)

content_field = TextField("content")
type_field = TextField("type")
fields_for_index = [content_field, type_field, md_embedding_field]

try:
    r.ft(INDEX_NAME).create_index(fields=fields_for_index)
    print(f"Index {INDEX_NAME} created successfully.")
except Exception as e:
    print(f"Error creating index {INDEX_NAME}: {e}")

def store_data(chunks, embeddings, data_type):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        key = f"{INDEX_NAME}:{data_type}:{i}"
        mapping_data = {
            "content": chunk,
            "type": data_type,
            "md_embedding": np.array(embedding, dtype=np.float32).tobytes()
        }

        r.hset(key, mapping=mapping_data)
        print(f"Stored {data_type} data with key: {key}")

store_data(text_chunks, text_embeddings, "text")
store_data(table_chunks, table_embeddings, "table")
store_data(image_descriptions, image_embeddings, "image")

import numpy as np
import json
from redis.commands.search.query import Query
# Search
# user_question = "工商银行2024年海外布局？"
user_question = "工商银行为什么被成为”宇宙行“？"

# Helper functions
def json_gpt(input: str):
    completion = client.chat.completions.create(
        model="qwen3:14b",
        messages=[
            {"role": "system", "content": "Output only valid JSON"},
            {"role": "user", "content": input},
        ],
        temperature=0.2,
    )

    text = completion.choices[0].message.content
    parsed = json.loads(text)

    return parsed


HA_INPUT = f"""
You are a financial report analysis assistant.
You have access to a search API that returns relevant sections from a financial report.
Generate a search query by extracting key words from the user's question.

User question: {user_question}
ddddda
Format: {{"searchQuery": "search query"}}
"""
query_str = json_gpt(HA_INPUT)["searchQuery"]
print(query_str)

query_embedding = client.embeddings.create(input=query_str, model="nomic-embed-text:latest", dimensions=1024, encoding_format="float")
query_vec = np.array(query_embedding.data[0].embedding, dtype=np.float32).tobytes()
# Prepare the query
k_nearest = 3  # Retrieve top 3 relevant chunks
query_base = (Query(f"*=>[KNN {k_nearest} @md_embedding $vec as score]").sort_by("score").return_fields("score", "content", "type").dialect(2))
query_param = {"vec": query_vec}
try:
    query_results = r.ft(INDEX_NAME).search(query_base, query_param).docs
    print(f"\nRetrieved {len(query_results)} results:")
    for i, doc in enumerate(query_results):
        print(f"\n--- Result {i+1} ---")
        print(f"Type: {doc.type}")
        print(f"Score: {doc.score}")
        print(f"Content (first 200 chars): {doc.content[:200]}...")
except Exception as e:
    print(f"Error executing vector query: {e}")
    query_results = []
# Prepare context for the LLM
context = "\n\n".join([doc.content for doc in query_results])

system_prompt = "你是一个金融报告分析专家。请根据搜索结果回答用户提问，注意，请务必首先依赖搜索结果，而不是你自己已有的知识。如果搜索结果不包含能够回答用户提问的信息，你可以说“抱歉，我无法回答这个问题”。"
messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": "用户提问：" + user_question},
            {"role": "user", "content": "搜索结果：" + context}]

response = client.chat.completions.create(
                    messages=messages,
                    model="qwen3:14b", #"qwen-max",
                    max_tokens=1000
                )
print(response.choices[0].message.content)



                                  