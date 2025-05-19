from io import BytesIO
from typing import List

import magic

from src.utils.constants import SUPPORTED_FORMATS
from src.utils.custom_exceptions import UnsupportedFileFormatError


def _get_file_format(file: BytesIO) -> str:
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


def validate_file_format(file: BytesIO) -> None:
    """
    Validate the file format of the given file is among the supported formats.

    Parameters
    ----------
    file: io.BytesIO
        File to validate.

    Raises
    ------
    UnsupportedFileFormatError
        If the file format is not supported.
    """
    file_format = _get_file_format(file=file)
    if file_format not in SUPPORTED_FORMATS:
        raise UnsupportedFileFormatError()


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
