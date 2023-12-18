"""
특정 파일들에서 번역된 결과를 저장하는 프로그램
src/samples/file01.mp3, src/samples/file02.mp3와 같이 음성파일이 존재한다면
src/file01.txt, src/file02.txt에 변환된 값을 저장함
"""
from transformers import pipeline
import librosa
import numpy
import os

transcriber = pipeline("automatic-speech-recognition",
                       model="openai/whisper-base",
                       generate_kwargs={"language" : "<|ko|>"},
                       chunk_length_s=24
                       )

def transcribe(audio_path, output_path):
    audio_data, sr = librosa.load(audio_path, sr=None)

    with open(output_path, "a") as file:
        file.write(transcriber({"sampling_rate":sr, "raw":audio_data})["text"])
        file.write('\n')
        
for file_name in os.listdir('src/samples'):
    audio_path = os.path.join('src/samples', file_name)
    output_path = file_name[:-3] + "txt"
    transcribe(audio_path, output_path)