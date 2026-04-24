# serve on http://127.0.0.1:7860/

from transformers import pipeline
from transformers.utils import logging
import gradio as gr

logging.set_verbosity_error()

# Load the speech recognition pipeline
# This is the same model used in your lesson.
asr = pipeline(
    task="automatic-speech-recognition",
    model="distil-whisper/distil-small.en",
)

def transcribe_audio(filepath: str) -> str:
    """
    Transcribe an audio file from either microphone input or uploaded file.
    Gradio provides a temporary file path when type='filepath'.
    """
    if not filepath:
        return "No audio file provided."

    try:
        output = asr(
            filepath,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=8,
        )
        return output["text"]
    except Exception as e:
        return f"Error during transcription: {e}"

with gr.Blocks(title="Speech Recognition App") as demo:
    gr.Markdown("# Automatic Speech Recognition")
    gr.Markdown("Transcribe microphone recordings or uploaded audio files.")

    with gr.Tab("Microphone"):
        mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Record audio")
        mic_output = gr.Textbox(label="Transcription", lines=6)
        mic_button = gr.Button("Transcribe")
        mic_button.click(fn=transcribe_audio, inputs=mic_input, outputs=mic_output)

    with gr.Tab("Upload File"):
        file_input = gr.Audio(sources=["upload"], type="filepath", label="Upload audio")
        file_output = gr.Textbox(label="Transcription", lines=6)
        file_button = gr.Button("Transcribe")
        file_button.click(fn=transcribe_audio, inputs=file_input, outputs=file_output)

if __name__ == "__main__":
    demo.launch()