---
title: Read-Aloud-AI
emoji: 🎙️️
sdk: gradio
sdk_version: 5.29.1
python_version: 3.10
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

## Poetry 1.8.4

```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.4
poetry config virtualenvs.in-project true
poetry install
```
