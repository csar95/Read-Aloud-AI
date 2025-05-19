---
title: Read-Aloud-AI
emoji: 🎙️️
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
suggested_hardware: cpu-basic
short_description: Turn PDFs into podcasts
models:
  - hexgrad/Kokoro-82M
tags:
  - Text-to-Speech
pinned: true
preload_from_hub:
  - hexgrad/Kokoro-82M kokoro-v1_0.pth
---

# Read-Aloud-AI

Turn any document into a podcast using the power of state-of-the-art Text-to-Speech (TTS) models.

This project takes written documents—like articles, reports, or notes—and transforms them into natural-sounding audio, making content easy to consume on the go.

## How Does Read-Aloud-AI Work?

The application converts PDF documents into podcast-style audio through the following main steps:

1. **PDF file upload and validation**  
  The user uploads a PDF file and selects the pages to convert. The system validates the file format and the specified pages.

2. **Text extraction and cleaning**  
  Text is extracted from the selected pages, with headers and footers automatically removed to ensure the content is clean and suitable for voice conversion.

3. **Text formatting with LLM**  
  The extracted text is sent to a language model (LLM, such as Gemini or GPT) to rewrite it in a more natural and voice-friendly format, inserting pauses where needed to enhance the listening experience.

4. **Text-to-Speech (TTS) conversion**  
  The formatted text is split into segments and converted into audio using an advanced TTS model. Configurable pauses are inserted between segments to simulate more natural speech.

5. **Generation and delivery of the final audio**  
  All audio segments are concatenated and delivered to the user as an audio file, ready to be listened to as a podcast.

This workflow is implemented in the [`generate_podcast_from_file`](src/inference.py) function, which orchestrates the entire process: from validation and text extraction, through LLM formatting, to final audio synthesis and assembly.

## Running the App with Poetry

To ensure you are using the same environment as the app, follow these steps to set up and run the project with [Poetry](https://python-poetry.org/):

```bash
# Install Poetry (version 1.8.4)
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.4

# Configure Poetry to create the virtual environment inside the project directory
poetry config virtualenvs.in-project true

# Install all dependencies as specified in pyproject.toml
poetry install
```

After installation, you can activate the environment and run the app:

```bash
poetry shell
python app.py
```

## Hugging Face Space Deployment

To deploy this app as a [Hugging Face Space](https://huggingface.co/spaces), two additional files are included:

- **requirements.txt**: Lists all Python dependencies needed to run the app in the Hugging Face environment.
- **packages.txt**: Specifies any system-level packages required by the app.

These files ensure the app is set up correctly and runs smoothly when hosted on Hugging Face Spaces.

To generate the `requirements.txt` file from your Poetry environment run the following command:

```bash
poetry export -f requirements.txt --without-hashes > requirements.txt
```