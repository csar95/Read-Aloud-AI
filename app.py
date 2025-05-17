from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import gradio as gr
from huggingface_hub import hf_hub_download
from kokoro import KModel

from src.inference import generate_podcast_from_file
from src.utils.constants import TTS_MODEL_REPO_ID


demo = gr.Interface(
    fn=generate_podcast_from_file,
    inputs=[
        gr.File(
            file_count="single",
            label="Upload a file",
            file_types=[".pdf"],
            type="binary",
        ),
        gr.Dropdown(
            choices=["am_liam", "am_puck"],  # FIXME: TAKE CHOICES FROM KOKORO
            value="am_liam",
            label="Voice",
            filterable=True,
            multiselect=False,
            allow_custom_value=False,
        ),
        gr.Slider(
            minimum=0.25,
            maximum=2,
            step=0.25,
            value=1,
            label="Speed of speech",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            value=0.3,
            label="Duration of pauses in the speech (seconds)",
        ),
        gr.Textbox(label="Gemini API Key", type="text", lines=1, max_lines=None),
        gr.Dropdown(
            choices=["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25"],
            value="gemini-2.5-flash-preview-04-17",
            label="Gemini model ID",
            filterable=True,
            multiselect=False,
            allow_custom_value=False,
        ),
    ],
    outputs=[
        gr.Audio(label="Podcast"),
    ],
)


if __name__ == "__main__":
    print("Downloading TTS model from Hugging Face Hub...")
    
    tts_model_path = hf_hub_download(
        repo_id=TTS_MODEL_REPO_ID,
        filename=KModel.MODEL_NAMES[TTS_MODEL_REPO_ID],
        force_download=False,  # Set to True to force redownload even if the file exists
    )

    demo.launch()
