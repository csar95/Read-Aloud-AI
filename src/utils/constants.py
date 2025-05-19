from typing import Annotated


DURATION_OF_ERROR_MESSAGE: Annotated[
    int, "Duration (in seconds) for which the error message is displayed"
] = 5
GEMINI_BASE_URL: Annotated[
    str, "Base URL for accessing the Gemini generative language API"
] = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_INFERENCE_RETRIES: Annotated[
    int, "Maximum number of retries for inference requests to the OpenAI API"
] = 3
OPENAI_API_KWARGS: Annotated[
    dict[str, str], "Keyword arguments for the OpenAI API client to be used when formatting the document text for TTS"
] = {
    "temperature": 0.0,
}
SAMPLE_RATE: Annotated[
    int, "Sample rate (in Hz) used for audio processing."
] = 24_000
SEND_REQUEST_TIMEOUT: Annotated[
    int, "Timeout for sending a request to the OpenAI API",
] = 60
SILENCE_KEYWORD: Annotated[
    str, "Keyword used to represent a pause when processing audio"
] = "[SILENCE]"
SUPPORTED_FORMATS: Annotated[
    list[str], "List of MIME types for document formats supported by the application"
] = [
    "application/pdf"
]
TIME_BETWEEN_RATE_LIMIT_RETRIES: Annotated[
    int,
    "Time (in seconds) to wait between retries when rate limit is hit based on the RPM limit",
] = 61
TTS_MODEL_REPO_ID: Annotated[
    str, "Repository ID for the text-to-speech (TTS) model"
] = "hexgrad/Kokoro-82M"
