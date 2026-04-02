import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import json


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream" : False
    })
    response = r.json()
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 


similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
#print(similarities)
top_results = 15
max_indx = similarities.argsort()[::-1][0:top_results]
#print(max_indx)
new_df = df.loc[max_indx] 
""" prompt = f'''{new_df[["title","number","start","end","text"]].to_json()} 
-----------------------------------
"{incoming_query}"
user asked this question related to the data that was provided like pdf's videos and more 
if the user asks unrelated question,tell hiim that you can only answer questions related to 
the data that was provided''' """
prompt = f"""
You are an intelligent and precise AI assistant designed to answer questions strictly based on provided context.

--------------------- CONTEXT ---------------------
{new_df[["title","number","start","end","text"]].to_json()}
---------------------------------------------------

User Question:
{incoming_query}

Instructions:
1. Answer ONLY using the information from the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not present in the context, respond with:
   "I don't know based on the provided data."
4. Be clear, concise, and structured.
5. If possible, summarize the answer in bullet points.
6. Do NOT hallucinate or make assumptions.

Answer:
"""
#print(new_df[["title", "number", "text"]])
response = inference(prompt)["response"]
print(response)
with open("response.txt","w") as f:
    f.write(response)