import html
import re

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

import pymupdf

from pypdfium2 import PdfDocument


def _replace_escape_characters(text: str, replacement: str = ""):
    # Define a regular expression to match escape characters
    escape_characters_pattern = re.compile(r'\\[abfnrtv\'"\\]|[\x00-\x1f\x7f-\x9f]')
    # Replace all escape characters with the specified replacement
    return escape_characters_pattern.sub(replacement, text)


def _find_common_elements(lines: List[str], page_count: int, threshold: float = 0.6):
    # We cannot filter if we don't have enough pages to find common elements
    if page_count < 3:
        return []
    lines_without_digits = [re.sub(r"\d+", "", line) for line in lines]
    counter = Counter(lines_without_digits)
    # If line appears in more than [threshold]% of the pages, it's probably a header or footer
    common = [line for line, count in counter.items() if count > page_count * threshold]
    # Return the original lines (with digits) that are common
    return [
        lines[line_id]
        for line_id, line_no_digits in enumerate(lines_without_digits)
        if line_no_digits in common
    ]


def detect_header_footer(
    document: PdfDocument, max_selected_lines: int = 5
) -> set[str]:
    """
    Detect header and footer lines in the PDF document. This function extracts the first
    and last lines of each page and finds the most common lines among all pages.

    Parameters
    ----------
    document: PdfDocument
        The PDF document to analyze.
    max_selected_lines: int
        The number of lines to select from the beginning and end of each page. Default
        is 5.

    Returns
    -------
    set: The set of header and footer lines found in the document.
    """
    first_last_lines = []
    for page_id in range(len(document)):
        page_text = document[page_id].get_textpage().get_text_range()
        page_lines = [line for line in page_text.splitlines() if line.strip() != ""]
        nonblank_lines = [
            line
            for line in page_lines
            if _replace_escape_characters(line).strip() != ""
        ]
        first_last_lines.extend(nonblank_lines[:max_selected_lines])
        first_last_lines.extend(nonblank_lines[-max_selected_lines:])

    header_footer_lines = _find_common_elements(
        first_last_lines, page_count=len(document)
    )
    return set(header_footer_lines)


def find_tables_in_pages(
    pdf_document: Union[BytesIO, str, Path],
    pages: List[int],
    strategy: str = "lines_strict",
) -> Dict[int, List[str]]:
    """
    Find tables in a specific page in the PDF file.

    Parameters
    ----------
    pdf_document: BytesIO, str or Path:
        The PDF document to search for tables.
    pages: List of int
        The list of pages (0-indexed) to search for tables.

    Returns
    -------
    Dict of List of str: List of tables found in each page in Markdown format.
    In case of an error in the PyMuPDF library, an empty list is returned.
    """
    try:
        if isinstance(pdf_document, (str, Path)):
            doc_aux = pymupdf.open(filename=pdf_document)
        else:
            try:
                file_stream = BytesIO(pdf_document.read())
                doc_aux = pymupdf.open(stream=file_stream)
            finally:
                pdf_document.seek(0)
    except Exception as exception:
        raise FileNotFoundError(exception)

    tables_by_page = dict()
    for page_id in pages:
        try:
            tables_in_page_id = doc_aux[page_id].find_tables(strategy=strategy).tables
        except Exception:
            tables_in_page_id = []
        tables_by_page[page_id] = [table.to_markdown() for table in tables_in_page_id]

    # From documentation: Release objects and space allocations associated with the
    # document. If created from a file, also closes filename (releasing control to the
    # OS). Explicitly closing a document is equivalent to deleting it, del doc, or
    # assigning it to something else like doc = None.
    doc_aux.close()

    return tables_by_page


def is_paragraph_completed(lines: List[str], line_id: int) -> bool:
    """
    Check if the paragraph is completed based on the current line and the next line.
    """
    return (
        # Last line
        line_id == len(lines) - 1
        # Next line is empty
        or lines[line_id + 1] == ""
        # Next line starts with an uppercase letter and current line ends with a period or colon
        or (lines[line_id + 1][0].isupper() and lines[line_id][-1] in [".", ":"])
        # Next line starts with a non-alphanumeric character
        or not lines[line_id + 1][0].isalnum()
        # Next line is part of an ordered list
        or re.match(r"^\d+[\.\)-]", lines[line_id + 1]) is not None
        or re.match(r"^[a-z][\.\)-]", lines[line_id + 1]) is not None
        # Last item of list
        or (
            lines[line_id].startswith("- ")
            or re.match(r"^\d+[\.\)-]", lines[line_id]) is not None
            or re.match(r"^[a-z][\.\)-]", lines[line_id]) is not None
        )
        and lines[line_id + 1][0].isupper()
    )


def decode_html_entities(text: str) -> str:
    # First, unescape &amp; and other named entities
    text = html.unescape(text)
    # Use regex to find all numeric entities and replace them with their character equivalents
    text = re.sub(r"&#(\d+);", lambda x: chr(int(x.group(1))), text)
    return text
