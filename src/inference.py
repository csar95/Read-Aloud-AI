from io import BytesIO
import json
from pathlib import Path
from typing import List

from huggingface_hub.constants import HF_HUB_CACHE
from kokoro import KModel, KPipeline
import gradio as gr
import magic
import numpy as np
from openai import OpenAI
from openai._exceptions import OpenAIError
import pypdfium2 as pdfium

from src.io_schemas.output_schemas import FormattedPageText
from src.io_schemas.prompts import FORMAT_TEXT_FOR_TTS
from src.openai_api_utils.controller import OpenAIAPIController
from src.pdf_reader.helpers import detect_header_footer
from src.utils.constants import (
    DURATION_OF_ERROR_MESSAGE,
    GEMINI_BASE_URL,
    OPENAI_API_KWARGS,
    SAMPLE_RATE,
    SILENCE_KEYWORD,
    SUPPORTED_FORMATS,
    TTS_MODEL_REPO_ID,
)
from src.utils.custom_exceptions import (
    OpenAIInvalidResponseFormatError,
    UnsupportedFileFormatError,
)


def setup_api_client(api_key: str, model_id: str) -> OpenAIAPIController:
    """
    Set up the OpenAI API client with the provided API key and initialize the controller
    for the specified model ID.

    Parameters
    ----------
    api_key : str
        The API key for the OpenAI API.
    model_id : str
        The model ID to be used for inference.

    Raises
    ------
    OpenAIError
        If the API key is invalid or if there is an issue while setting up the client.

    Returns
    -------
    OpenAIAPIController
        An instance of the OpenAIAPIController initialized with the OpenAI client and
        model ID.
    """
    if not api_key:
        raise OpenAIError("API key is required.")

    try:
        client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
        print(client.models.list().to_dict())
    except Exception:
        raise OpenAIError("Invalid API key or unable to connect to the OpenAI API.")

    openai_api_ctrl = OpenAIAPIController(openai_client=client, model_name=model_id)
    return openai_api_ctrl


def get_file_format(file: BytesIO) -> str:
    """
    Get the file format of the given file.

    Parameters
    ----------
    file: io.BytesIO
        File to get the format from.

    Returns
    -------
    str
        File format.
    """
    try:
        content = file.read()
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(content)
    finally:
        file.seek(0)

    return file_type


def extract_text_from_pdf(pdf_file: BytesIO, pages: List[int] = None) -> List[str]:
    """
    Extract text from a PDF file, removing header and footer lines.

    Parameters
    ----------
    pdf_file : io.BytesIO
        The PDF file to extract text from.
    pages : list of int, optional
        List of page numbers (0-indexed) to extract text from. If None, all pages will
        be processed. Default is None.

    Returns
    -------
    list of str
        List of strings, each representing the text extracted from a page of the PDF.
    """

    pdf = pdfium.PdfDocument(pdf_file)
    print(f"Length of PDF: {len(pdf)} pages")

    header_footer_lines = detect_header_footer(document=pdf)

    text_from_pages = []
    for page_id in pages or range(len(pdf)):
        # It seems that the package "pypdfium2" separates lines by "\r\n" by default
        page_text = pdf[page_id].get_textpage().get_text_bounded()

        # Remove lines contained in header/footer
        page_text_without_header_footer = "\n".join(
            line
            for line in page_text.splitlines()
            if line.strip() not in header_footer_lines
        )

        text_from_pages.append(page_text_without_header_footer)

    return text_from_pages


def format_text_for_tts(
    openai_api_controller: OpenAIAPIController, text_chunks: List[str]
) -> str:
    """
    Format text chunks into clean, natural-language text suitable for Text-to-Speech
    (TTS) using an LLM model.

    Parameters
    ----------
    openai_api_controller : OpenAIAPIController
        The OpenAI API controller to use for sending requests to the LLM model.
    text_chunks : list of str
        List of text chunks to format.

    Raises
    ------
    OpenAIInvalidResponseFormatError
        If the response from the OpenAI API does not match the expected format.

    Returns
    -------
    str
        The formatted text as a single string.
    """
    formatted_document_text = ""
    for page_id, page_text in enumerate(text_chunks):
        print(("-------------------------------------------------------------------"))
        print(f"Processing page {page_id + 1}/{len(text_chunks)}")

        # Get input texts that are needed to build the prompt
        previous_fragment = (
            f"... {formatted_document_text[-100:]}" if formatted_document_text else ""
        )
        current_page = page_text
        next_preview = (
            text_chunks[page_id + 1] if page_id + 1 < len(text_chunks) else ""
        )

        # Build the prompt object as the OpenAI API controller expects
        prompt = {
            "system_msg": FORMAT_TEXT_FOR_TTS.system_msg.format(
                silence_keyword=SILENCE_KEYWORD,
            ),
            "user_msg": FORMAT_TEXT_FOR_TTS.user_msg.format(
                previous_fragment=previous_fragment,
                current_page=current_page,
                next_preview=next_preview,
            ),
        }

        # Send the request to the OpenAI/Gemini API
        chat_completion, elapsed_time_s, retries_taken = (
            openai_api_controller.send_request(
                prompt=prompt,
                response_format=FORMAT_TEXT_FOR_TTS.output_json,
                **OPENAI_API_KWARGS,
            )
        )
        num_attempts = retries_taken + 1
        print(
            f"Received response from OpenAI API. Response: {chat_completion}\n"
            f"num_attempts: {num_attempts}\n"
            f"response_time (s): {elapsed_time_s:.2f}"
        )

        # Validate response format. Raises custom exception if the response does not match the `FormattedPageText` schema
        response_msg = chat_completion.choices[0].message.content
        try:
            formatted_page_text = FormattedPageText(**json.loads(response_msg))
        except Exception as e:
            raise OpenAIInvalidResponseFormatError()

        formatted_document_text += f" {formatted_page_text.text}"

    return formatted_document_text


def chunk_text(text: str, max_words: int = 50) -> List[str]:
    """
    Splits the text into chunks formed by sentences, ensuring that each chunk does not
    exceed the specified number of words.

    Parameters
    ----------
    text : str
        The text to be split into chunks.
    max_words : int
        The maximum number of words allowed in each chunk.

    Returns
    -------
    List[str]
        A list of text chunks, each containing a maximum of `max_words` words.
    """
    small_chunks_for_tts = []
    current_chunk = ""
    # Split the text into sentences using ". " as a delimiter and filter out empty strings
    for sentence in filter(lambda x: x != "", map(str.strip, text.split(". "))):

        current_chunk_words = current_chunk.split()
        sentence_words = sentence.split()

        if (
            current_chunk != ""
            and len(current_chunk_words) + len(sentence_words) > max_words
        ):
            small_chunks_for_tts.append(current_chunk.rstrip())
            current_chunk = ""

        current_chunk += sentence + (". " if sentence[-1] != "." else "")

    if current_chunk:
        small_chunks_for_tts.append(current_chunk.rstrip())

    return small_chunks_for_tts


def text_to_speech(
    text: str, voice: str, speed: float, duration_of_pauses: float
) -> np.ndarray:
    tts_model_path = str(next(Path(HF_HUB_CACHE).rglob("kokoro-v1_0.pth")))
    tts_model = KModel(repo_id=TTS_MODEL_REPO_ID, model=tts_model_path)
    pipeline = KPipeline(
        lang_code="a",
        repo_id=TTS_MODEL_REPO_ID,
        model=tts_model,
        device="cpu",  # FIXME: GET DEVICE DEPENDING ON GPU AVAILABILITY
    )

    audio_chunks = []
    silence = np.zeros(int(SAMPLE_RATE * duration_of_pauses), dtype=np.float32)

    text_split_by_pauses = filter(
        lambda x: x != "", map(str.strip, text.split(SILENCE_KEYWORD))
    )
    for text_chunk_id, text_between_pauses in enumerate(text_split_by_pauses):

        audios_from_text_between_pauses = []
        text_chunks = chunk_text(text=text_between_pauses)
        for small_text_chunk in text_chunks:

            audio_generator = pipeline(text=small_text_chunk, voice=voice, speed=speed)
            for graphemes, phonemes, audio_chunk in audio_generator:
                print(
                    f"++++ Processing audio chunk\n"
                    f" Number of words: {len(graphemes.split())}\n"
                    f" Graphemes: {graphemes}\n"
                    f" Phonemes: {phonemes}"
                )
                audios_from_text_between_pauses.append(audio_chunk)

        audios_from_text_between_pauses = np.concatenate(
            audios_from_text_between_pauses
        )
        if text_chunk_id > 0:
            audios_from_text_between_pauses = np.concatenate(
                [silence, audios_from_text_between_pauses]
            )

        audio_chunks.append(audios_from_text_between_pauses)

    audio = np.concatenate(audio_chunks)
    return audio


def generate_podcast_from_file(
    file, voice, speed, duration_of_pauses, gemini_api_key, gemini_model_id
) -> np.ndarray:
    try:
        openai_api_ctrl = setup_api_client(
            api_key=gemini_api_key, model_id=gemini_model_id
        )

        byte_stream = BytesIO(file)
        file_format = get_file_format(file=byte_stream)
        assert file_format in SUPPORTED_FORMATS, str(UnsupportedFileFormatError())

        text_from_pages = extract_text_from_pdf(
            pdf_file=byte_stream, pages=list(range(4))
        )  # FIXME: REMOVE HARDCODED PAGES

        formatted_text = format_text_for_tts(
            openai_api_controller=openai_api_ctrl, text_chunks=text_from_pages
        )

        audio = text_to_speech(
            text=formatted_text,
            voice=voice,
            speed=speed,
            duration_of_pauses=duration_of_pauses,
        )

        return SAMPLE_RATE, audio

    except OpenAIError as e:
        print(f"ERROR: {e}")
        raise gr.Error(
            message="Gemini API client is not available. Please check your API key and model ID.",
            duration=DURATION_OF_ERROR_MESSAGE,
        )

    except Exception as e:
        print(f"ERROR: {e}")
        raise gr.Error(message=str(e), duration=DURATION_OF_ERROR_MESSAGE)
