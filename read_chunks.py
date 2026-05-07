import requests
import os
import json
import pandas as pd
import joblib


MODEL_NAME = "nomic-embed-text"

def create_embedding(text_list):

    text_list = [t.strip() for t in text_list if t and t.strip()]

    if not text_list:
        return []

    try:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": MODEL_NAME,
                "input": text_list
            }
        ).json()


        if "embeddings" not in response:
            print(" ERROR FROM OLLAMA:", response)
            return []

        return response["embeddings"]

    except Exception as e:
        print(" REQUEST FAILED:", e)
        return []


#BATCH FUNCTION
def batch(lst, size=128):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]



jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"\n🚀 Creating Embeddings for {json_file}")


    texts = [c["text"].strip() for c in content["chunks"] if c["text"].strip()]

    all_embeddings = []


    for b in batch(texts, 128):
        print(f"Processing batch of {len(b)} texts")
        emb = create_embedding(b)
        all_embeddings.extend(emb)


    if len(all_embeddings) != len(texts):
        print("⚠️ Mismatch in embeddings count, skipping file")
        continue


    idx = 0
    for chunk in content["chunks"]:
        if not chunk["text"].strip():
            continue

        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = all_embeddings[idx]

        chunk_id += 1
        idx += 1
        my_dicts.append(chunk)


df = pd.DataFrame.from_records(my_dicts)

print("\n✅ DONE")

joblib.dump(df, 'embeddings.joblib')