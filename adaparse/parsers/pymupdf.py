"""The PyMuPDF PDF parser."""

from __future__ import annotations

import re
from typing import Any
from typing import Literal

import pymupdf

from adaparse.parsers.base import BaseParser
from adaparse.parsers.base import BaseParserConfig
from adaparse.utils import exception_handler

__all__ = [
    'PyMuPDFParser',
    'PyMuPDFParserConfig',
]

class PyMuPDFParserConfig(BaseParserConfig):
    """Settings for the PyMuPDF-PDF parser."""

    # The name of the parser.
    name: Literal['pymupdf'] = 'pymupdf'  # type: ignore[assignment]


class PyMuPDFParser(BaseParser):
    """Interface for the PyMuPDF PDF parser."""

    def __init__(self, config: PyMuPDFParserConfig) -> None:
        """Initialize the marker parser."""
        self.config = config
        self.abstract_threshold = 580

    def extract_doi_info(self, input_str: str) -> str:
        """Extract doi from PyMUPDF metadata entry (if present)."""
        match = re.search(r'(doi:\s*|doi\.org/)(\S+)', input_str)
        return match.group(2) if match else ''

    @exception_handler(default_return=None)
    def parse_pdf(self,
                  pdf_path: str) -> tuple[str, dict[str, Any]] | None:
        """Parse a PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.
        pagewise_flag: bool
             Indicates if the page texts are returned as a list[str] per document, or merged into single str.

        Returns
        -------
        tuple[str, dict[str, str]] | None
            A tuple containing the full text of the PDF and the metadata
            extracted from the PDF. If parsing fails, return None.
        """
        # open pdf
        with pymupdf.open(pdf_path) as doc:
            # scrape text
            page_texts_list = []
            # track char page indices
            cumm_idx = 0
            page_indices = [0]

            # loop pages
            for page in doc:
                # - page's text
                page_txt = page.get_text()
                page_texts_list.append(page_txt)
                # - char indices
                cumm_idx += len(page_txt) + len('\n')
                page_indices.append(cumm_idx)

            # remove trailing index
            page_indices = page_indices[:-1]

            full_text = '\n'.join(page_texts_list)

            # Get first page (as a proxy for `abstract`)
            first_page_text = page_texts_list[0] if len(page_texts_list) > 0 else ''

            # metadata (available to PyMuPDF)
            title = doc.metadata.get('title', '')
            authors = doc.metadata.get('author', '')
            creationdate = doc.metadata.get('creationDate', '')
            keywords = doc.metadata.get('keywords', '')
            doi = self.extract_doi_info(doc.metadata.get('subject', ''))
            prod = doc.metadata.get('producer', '')
            form = doc.metadata.get('format', '')
            abstract = (
                doc.metadata.get('subject', '')
                if len(doc.metadata.get('subject', '')) > self.abstract_threshold
                else ''
            )

            # assemble the metadata
            out_meta = {
                'title': title,
                'authors': authors,
                'creationdate': creationdate,
                'keywords': keywords,
                'doi': doi,
                'producer': prod,
                'format': form,
                'abstract': abstract,
                'first_page': first_page_text,
                'page_text_list': page_texts_list,
                'page_char_idx': page_indices,
            }

            # full text & metadata entries
            return full_text, out_meta

    @exception_handler(default_return=None)
    def parse(self,
              pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""
        # list[str] (by_doc) or list[list[str]] (by_page)
        documents = []

        # Process each PDF
        for pdf_file in pdf_files:
            # Parse the PDF
            output = self.parse_pdf(pdf_file)

            # Check if the PDF was parsed successfully
            if output is None:
                print(f'Error: Failed to parse {pdf_file}')
                continue

            # Unpack the output
            text, metadata = output

            # Setup the document fields to be stored
            document = {
                'text': text,
                'path': str(pdf_file),
                'metadata': metadata,
                'parser': self.config.name,
            }

            # text & meta
            documents.append(document)

        return documents
