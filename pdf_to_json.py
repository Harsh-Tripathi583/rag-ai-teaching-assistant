from pypdf import PdfReader
import os
import json

Documents = os.listdir("Documents")

for idx, doc in enumerate(Documents):
    print(f"Processing : {doc}")

    reader = PdfReader(f"Documents/{doc}")

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # Clean text
    text = text.replace("\n", " ")

    # Chunking
    chunk_size = 500
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    # Extract PDF name
    Video_Title = doc.split(".")[0]

    # Create SAME format as video
    formatted_chunks = []

    for i, chunk in enumerate(chunks):
        formatted_chunks.append({
            "Video_Number": f"pdf_{idx}",   # unique id per PDF
            "Video_Title": Video_Title,
            "start": i * chunk_size,       # fake start
            "end": (i+1) * chunk_size,     # fake end
            "text": chunk
        })


    final_output = {
        "chunks": formatted_chunks,
        "text": text
    }


    output_file = doc.replace(".pdf", ".json")


    with open(f"jsons/{output_file}", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

print("All PDFs processed ✅")