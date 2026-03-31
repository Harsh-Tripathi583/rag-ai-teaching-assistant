import whisper
import os
import json

model = whisper.load_model("medium").to("cuda")

Audios = os.listdir("Audios")

for Audio in Audios:
    Video_Number = Audio.split("-")[0]
    Video_Title = Audio.split("-")[1].split(".")[0]
    
    result = model.transcribe(audio=f"Audios/{Audio}",task="translate",word_timestamps=False)

    chunks = []
    for segment in result["segments"]:
        chunks.append({"Video_Number" : Video_Number,"Video_Title": Video_Title,"start":segment["start"] , "end": segment["end"] , "text": segment["text"]})
    
    chunks_with_metadata = {"chunks":chunks , "text":result["text"]}
    output_file = Audio.replace(".mp3", ".json")
    with open(f"jsons/{output_file}","w") as f:
        json.dump(chunks_with_metadata,f)
    break


print("All files processed ✅")