import re

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pypdfium2 as pdfium

from src.pdf_reader.helpers import (
    decode_html_entities,
    detect_header_footer,
    find_tables_in_pages,
    is_paragraph_completed,
)


class PDFReaderController:
    def __init__(self, pdf_document: Union[BytesIO, str, Path]):
        self.pdf_document = pdf_document
        try:
            if isinstance(pdf_document, (str, Path)):
                self.document = pdfium.PdfDocument(pdf_document)
            else:
                try:
                    file_stream = BytesIO(pdf_document.read())
                    self.document = pdfium.PdfDocument(file_stream)
                finally:
                    pdf_document.seek(0)
        except Exception as exception:
            raise FileNotFoundError(exception)

    def extract_formatted_text(
        self,
        pages: Optional[List[int]] = None,
        separator_between_pages: str = "\n",
        ignore_header_footer: bool = True,
    ) -> str:
        """
        Extract the raw text from the PDF document. This function concatenates the text
        of all pages into a single string, removes header and footer lines, and marks
        the lines that belong to tables.

        After extracting the raw text, the function formats the text by adding tables in
        Markdown format and joining sentences that belong to the same paragraph.

        Parameters
        ----------
        pages: List of int
            0-indexed list of page ids to extract text from. Default is None. If None,
            this function will extract text from all pages.
        separator_between_pages: str
            The separator to use between pages. Default is "\n".
        ignore_header_footer: bool
            Whether to ignore header and footer lines or not. Default is True.

        Returns
        -------
        pdf_text: str
            The formatted text of the document between the specified pages.
            If an error occurs, an empty string is returned.
        """
        try:
            header_footer_lines = (
                detect_header_footer(document=self.document)
                if ignore_header_footer
                else set()
            )

            markdown_tables_in_pages = find_tables_in_pages(
                pdf_document=self.pdf_document,
                pages=pages or range(len(self.document)),
            )

            # Extract the text of each page and concatenate them
            pdf_text = ""
            for page_id in pages or range(len(self.document)):
                # It seems that the package "pypdfium2" separates lines by "\r\n" by default
                page_text = self.document[page_id].get_textpage().get_text_bounded()

                # Split the text into lines removing the ones contained in the header and footer
                page_lines = [
                    line
                    for line in page_text.splitlines()
                    if line.strip() not in header_footer_lines
                ]

                # Identify the blocks of lines corresponding to each table in the page
                lines_ids_w_table_content = []
                for line_id, line in enumerate(page_lines):
                    for table_id, table in enumerate(markdown_tables_in_pages[page_id]):
                        if all(
                            [
                                word in decode_html_entities(table)
                                for word in line.split()
                            ]
                        ):
                            page_lines[line_id] = f"[PAGE_{page_id}_TABLE_{table_id}]"
                            lines_ids_w_table_content.append(line_id)
                            break

                # Naive correction of the lines that refer to other tables or no table at all
                seen_full_tables = []
                for line_id in lines_ids_w_table_content:
                    current_line = page_lines[line_id]
                    prev_line = page_lines[line_id - 1] if line_id > 0 else None

                    if (
                        prev_line is not None
                        and re.fullmatch(
                            pattern=r"\[PAGE_\d+_TABLE_\d+\]", string=prev_line
                        )
                        and prev_line != current_line
                    ):
                        # If the previous line is a table reference and the current line is not the same table,
                        # assume previous table is complete
                        if current_line not in seen_full_tables:
                            seen_full_tables.append(prev_line)
                        # If not, we assume that the current line is part of the same table as the previous line
                        else:
                            page_lines[line_id] = prev_line

                # Remove duplicate lines that refer to the same table
                lines_ids_to_remove = []
                for i, line_id in enumerate(lines_ids_w_table_content):
                    if i == 0:
                        seen_full_tables = [page_lines[line_id]]
                    elif page_lines[line_id] not in seen_full_tables:
                        seen_full_tables.append(page_lines[line_id])
                    else:
                        lines_ids_to_remove.append(line_id)
                page_lines = list(np.delete(page_lines, lines_ids_to_remove))

                # Put the lines back together
                page_text = "\n".join(page_lines)

                # Format the text of the page
                page_text = self.format_text(
                    text=page_text, markdown_tables_in_pages=markdown_tables_in_pages
                )

                pdf_text += (
                    (page_text + separator_between_pages)
                    if page_id + 1 < len(self.document)
                    else page_text
                )

            return pdf_text.strip()

        except Exception:
            return ""

    @staticmethod
    def format_text(text: str, markdown_tables_in_pages: List[List[str]]) -> str:
        """
        Format the text of the document by adding tables in Markdown format and joining
        sentences that belong to the same paragraph.

        Parameters
        ----------
        text: str
            The text of the document.
        markdown_tables_in_pages: List of lists of str
            The list of tables in Markdown format found in each page.

        Returns
        -------
        str: The formatted text of the document.
        """
        formatted_text, paragraph = [], []
        lines = [line for line in text.splitlines() if line.strip() != ""]

        for line_id, line in enumerate(lines):
            # If the line is a table reference, add the table in Markdown format
            if re.fullmatch(pattern=r"\[PAGE_\d+_TABLE_\d+\]", string=line):
                matches = re.findall(r"\[PAGE_(\d+)_TABLE_(\d+)\]", line)
                if matches:
                    try:
                        page_id, table_id = matches[0]
                        markdown_table = markdown_tables_in_pages[int(page_id)][
                            int(table_id)
                        ]
                        formatted_text.append(markdown_table.strip("\n"))
                    except Exception:  # E.g., IndexError
                        # The markdown table is omitted from the formatted text
                        pass
                    continue
            # Otherwise, add the line as a new sentence to the paragraph
            else:
                paragraph.append(line)

            if is_paragraph_completed(lines=lines, line_id=line_id):
                formatted_text.append(" ".join(paragraph))
                paragraph = []

        return "\n".join(formatted_text)
