import requests
import os
import json

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model" : "bge-m3",
        "input" : text_list
    })
    embeddings = r.json()["embeddings"]

    return embeddings


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}","r") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}🔃")
    embeddings = create_embedding([c["text"]for c in content["chunks"]])
    print(f" Embeddings Created  for {json_file}✅")

    for i,chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk_id+=1
        chunk["embeddings"] = embeddings[i]
        my_dicts.append(chunk)
    print(f"Chunks Stored in my_dicts ✅")

