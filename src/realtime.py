"""
gradio Interface를 사용해 Realtime-STT를 구현한 프로그램
실행 시키고, 링크에 들어가서 마이크를 작동시킨 후
음성을 입력하면 그 변환된 결과값이 src/realtime_result/result.txt에 저장됌
"""
import numpy as np
import gradio as gr
from transformers import pipeline

output_file_path = "src/realtime_result/result.txt"

transcriber = pipeline("automatic-speech-recognition",
                       model="openai/whisper-base",
                       generate_kwargs={"language" : "<|ko|>"},
                       )

def transcribe(stream, texts, new_chunk):
    global output_file_path
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if not stream:
        stream = [y]
        texts = [transcriber({"sampling_rate":sr, "raw":y})["text"]]
    elif len(stream[-1]) + len(y) < 24 * sr:
        stream[-1] = np.concatenate([stream[-1], y])
        texts[-1] = transcriber({"sampling_rate":sr, "raw":stream[-1]})["text"]
    else:
        stream.append(y)
        texts.append(transcriber({"sampling_rate":sr, "raw":y})["text"])

    with open(output_file_path, "w") as file:
        for text in texts:
            file.write(text)
            file.write('\n')

    return stream, texts

with gr.Blocks() as demo:
    audio_source = gr.Audio(sources="microphone", streaming=True)
    saved_streams = gr.State([])
    saved_texts = gr.State([])
    
    audio_source.change(
        transcribe,
        inputs=[saved_streams, saved_texts, audio_source],
        outputs=[saved_streams, saved_texts]
    )
    
demo.launch()