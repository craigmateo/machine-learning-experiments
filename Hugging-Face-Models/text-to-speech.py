# serve on http://127.0.0.1:7860

from transformers import pipeline
from transformers.utils import logging
import gradio as gr
import soundfile as sf
import tempfile
import os

logging.set_verbosity_error()

# Load the text-to-speech pipeline
narrator = pipeline(
    "text-to-speech",
    model="kakao-enterprise/vits-ljs"
)

def generate_speech(text: str):
    """
    Generate speech audio from text and return a file path
    that Gradio can play.
    """
    if not text or not text.strip():
        return None, "Please enter some text."

    try:
        result = narrator(text)

        audio = result["audio"][0]
        sampling_rate = result["sampling_rate"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            temp_path = tmp_file.name

        sf.write(temp_path, audio, sampling_rate)

        return temp_path, "Done."
    except Exception as e:
        return None, f"Error: {e}"

with gr.Blocks(title="Text to Speech") as demo:
    gr.Markdown("# Text to Speech")
    gr.Markdown("Enter text and generate spoken audio.")

    text_input = gr.Textbox(
        label="Input Text",
        lines=8,
        placeholder="Type something here..."
    )

    generate_button = gr.Button("Generate Speech")
    audio_output = gr.Audio(label="Generated Audio", type="filepath")
    status_output = gr.Textbox(label="Status", lines=1)

    generate_button.click(
        fn=generate_speech,
        inputs=text_input,
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()