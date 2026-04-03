import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import json

#.venv\Scripts\activate
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
{"\n\n".join(new_df["text"].tolist())}
---------------------------------------------------

User Question:
{incoming_query}

Instructions:
- Answer using the context.
- Answer the question naturally using the provided context.Do not mention "context" in your answer.
- If incomplete, say "Based on my general knowledge..." and continue.
- Be clear and structured.

Answer:
"""
#print(new_df[["title", "number", "text"]])
response = inference(prompt)["response"]
print(response)
with open("response.txt","w") as f:
    f.write(response)