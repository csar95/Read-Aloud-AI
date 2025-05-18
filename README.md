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