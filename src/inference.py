from io import BytesIO
import json
from typing import List

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
from src.tts.controller import TTSModelClient
from src.utils.constants import (
    DURATION_OF_ERROR_MESSAGE,
    GEMINI_BASE_URL,
    OPENAI_API_KWARGS,
    SAMPLE_RATE,
    SILENCE_KEYWORD,
    SUPPORTED_FORMATS,
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

    Raises
    ------
    IndexError
        If the specified page number is out of range.

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


def convert_text_to_speech(
    client: TTSModelClient,
    text: str,
    voice: str,
    speed: float,
    duration_of_pauses: float,
) -> np.ndarray:
    """
    Converts text to speech using the TTS model. To do so, it splits the text into
    chunks based on the "silence" keyword, and generates audio for each chunk. This is
    done to add pauses in the speech so it sounds more natural.

    Parameters
    ----------
    client : TTSModelClient
        The TTS model client to use for generating audio.
    text : str
        The text to be read by the TTS model.
    voice : str
        The voice to be used for TTS.
    speed : float
        The speed of speech.
    duration_of_pauses : float
        The duration of pauses in the speech (in seconds).

    Returns
    -------
    np.ndarray
        The generated audio as a NumPy array.
    """
    audio_chunks = []
    silence = np.zeros(int(SAMPLE_RATE * duration_of_pauses), dtype=np.float32)

    text_split_by_pauses = filter(
        lambda x: x != "", map(str.strip, text.split(SILENCE_KEYWORD))
    )
    for text_chunk_id, text_between_pauses in enumerate(text_split_by_pauses):

        audios_from_text_between_pauses = client.text_to_speech(
            text=text_between_pauses, voice=voice, speed=speed
        )

        if text_chunk_id > 0:
            audios_from_text_between_pauses = np.concatenate(
                [silence, audios_from_text_between_pauses]
            )

        audio_chunks.append(audios_from_text_between_pauses)

    audio = np.concatenate(audio_chunks)
    return audio


def validate_input_pages(pages: str) -> List[int]:
    """
    Validate the input pages string and convert it to a 0-indexed list of integers.

    Notes
    -----
    The input can be a single page number (e.g., "1"), a list of pages (e.g., "1,2,3"),
    or a range of pages (e.g., "1-3").

    Parameters
    ----------
    pages : str
        The input string representing the pages to be processed.

    Raises
    ------
    ValueError
        If input cannot be converted to a list of integers.
    Exception
        If the input format is invalid or if both a list and a range of pages are
        provided.

    Returns
    -------
    list of int
        A list of integers representing the page numbers (0-indexed) to be processed.
    """
    pages = pages.strip()
    if not pages:
        return None
    elif "," in pages and "-" in pages:
        raise Exception("Cannot mix comma-separated and range formats.")
    elif "," in pages:
        page_numbers = [int(x) for x in filter(lambda x: x != "", map(str.strip, pages.split(",")))]
        if any(p <= 0 for p in page_numbers):
            raise Exception("Page numbers must be positive integers.")
        return [p - 1 for p in page_numbers]
    elif "-" in pages:
        start, end = map(int, filter(lambda x: x != "", map(str.strip, pages.split("-"))))
        if start <= 0 or end <= 0:
            raise Exception("Page numbers must be positive integers.")
        if start >= end:
            raise Exception("Start page cannot be greater than or equal to end page.")
        return list(range(start - 1, end))
    else:
        page_number = int(pages)
        if page_number <= 0:
            raise Exception("Page number must be a positive integer.")
        return [page_number - 1]


def generate_podcast_from_file(
    file, pages, voice, speed, duration_of_pauses, gemini_api_key, gemini_model_id
) -> np.ndarray:
    try:
        # Setup OpenAI API client
        openai_api_ctrl = setup_api_client(
            api_key=gemini_api_key, model_id=gemini_model_id
        )

        # Setup TTS model
        tts_client = TTSModelClient()

        # Check if file format is supported
        byte_stream = BytesIO(file)
        file_format = get_file_format(file=byte_stream)
        assert file_format in SUPPORTED_FORMATS, str(UnsupportedFileFormatError())

        # Validate input pages
        pages = validate_input_pages(pages=pages)

        text_from_pages = extract_text_from_pdf(pdf_file=byte_stream, pages=pages)

        formatted_text = format_text_for_tts(
            openai_api_controller=openai_api_ctrl, text_chunks=text_from_pages
        )

        audio = convert_text_to_speech(
            client=tts_client,
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
