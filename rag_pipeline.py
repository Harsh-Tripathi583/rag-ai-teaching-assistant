import requests
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import json

#.venv\Scripts\activate

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text_list
        }
    )

    return r.json()["embeddings"]


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]


df = joblib.load("embeddings.joblib")


def get_answer(incoming_query):

    question_embedding = create_embedding([incoming_query])[0]

    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()

    top_results = 7
    max_indx = similarities.argsort()[::-1][:top_results]

    new_df = df.loc[max_indx]

    prompt = f"""
You are an intelligent AI teaching assistant.

---------------- CONTEXT ----------------
{"\n\n".join(new_df["text"].tolist())}
-----------------------------------------

User Question:
{incoming_query}

Instructions:
- Answer only from the provided context.
- If answer is not present, say:
"I could not find this information in the provided study material."
- Be clear and structured.

Answer:
"""

    response = inference(prompt)

    return response