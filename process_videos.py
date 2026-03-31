#Converts mp4(Videos) to mp3(audios)
import os
import subprocess
files = sorted(os.listdir("Videos"))

for file in files:
    #print(file)
    video_no = file.split("_")[1].split(".")[0]
    video_title = file.split("_")[0]
    print(video_no,video_title)
    subprocess.run(["ffmpeg","-i",f"Videos/{file}",f"Audios/{video_no}-{video_title}.mp3"])
