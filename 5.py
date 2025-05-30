from redis import Redis
from redis.commands.search.field import VectorField, TextField
import numpy as np

r = Redis()


INDEX_NAME = "PDFData_VLforTable"
VECTOR_DIM = 1024
DISTANCE_METRIC = "COSINE"

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
# 写入一个键值对
key = f"1234565676"
mapping_data = {
    "md_embedding": np.array([10., 20, 30]).tobytes(),
    "content": "chunk",
    "type": "text"
}
r.set("mykey", "myvalue")