"""The Nougat PDF parser."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal

import pymupdf

from pydantic import field_validator

from adaparse.parsers.base import BaseParser
from adaparse.parsers.base import BaseParserConfig
from adaparse.parsers.device_utils import build_doc_and_indices
from adaparse.parsers.device_utils import move_to_custom_device
from adaparse.parsers.device_utils import resolve_device # new
from adaparse.parsers.nougat_inference_utils import prepare_input_sc

from adaparse.utils import exception_handler
from adaparse.utils import setup_logging


__all__ = [
    'NougatParser',
    'NougatParserConfig',
]
class NougatParserConfig(BaseParserConfig):
    """Settings for the marker PDF parser."""

    # The name of the parser.
    name: Literal['nougat'] = 'nougat'  # type: ignore[assignment]
    # The batch size for the parser (10 is the max that fits in an A100).
    batchsize: int = 10
    # The number of workers to use for dataloading.
    num_workers: int = 1
    # The Number of batches loaded in advance by each worker
    prefetch_factor: int = 4
    # The path to the Nougat model checkpoint.
    checkpoint: Path
    # The directory to write optional mmd outputs along with jsonls.
    mmd_out: Path | None = None
    # Override pre-existing parsed outputs.
    recompute: bool = False
    # Fill-in of failed/empty pages via PyMuPDF (False implies pure Nougat)
    fill_missing_pages: bool = False
    # Use float32 instead of bfloat32.
    full_precision: bool = False
    # Whether to format the output as markdown.
    markdown: bool = True
    # Skip if the model falls in repetition.
    skipping: bool = True
    # The directory to write the logs to.
    nougat_logs_path: Path

    @field_validator('mmd_out')
    @classmethod
    def validate_mmd_out_is_dir(cls, value: Path | None) -> Path | None:
        """Create the output directory if it does not exist."""
        if value is not None:
            value.mkdir(exist_ok=True, parents=True)
        return value

    @field_validator('checkpoint')
    @classmethod
    def validate_ckpt_path_exists(cls, value: Path) -> Path:
        """Check if the directory exists."""
        if not value.exists():
            from nougat.utils.checkpoint import get_checkpoint

            print(
                'Checkpoint not found in the directory you specified. '
                'Downloading base model from the internet instead.'
            )
            value = get_checkpoint(value, model_tag='0.1.0-base')
        return value


class NougatParser(BaseParser):
    """Warmstart interface for the marker PDF parser.

    Initialization loads the Nougat models into memory and registers them in a
    global registry unique to the current process. This ensures that the models
    are only loaded once per worker process (i.e., we warmstart the models)
    """

    def __init__(self, config: NougatParserConfig) -> None:
        """Initialize the marker parser."""
        import torch
        from nougat import NougatModel                  # nougat lib
        # from nougat_parser.model import NougatModel   # own implementation

        self.config = config
        self.model = NougatModel.from_pretrained(config.checkpoint)
        self.model.eval()

        # move model
        self.model = move_to_custom_device(
            self.model,
            bf16=not self.config.full_precision,
        )
        # compile model
        try:
            self.model = torch.compile(self.model, fullgraph=True)
        except Exception:
            # fall back silently; compilation is optional
            self.logger = setup_logging('[WARNING] Failed to compile model', config.nougat_logs_path)

        self.logger = setup_logging('adaparse_nougat', config.nougat_logs_path)

        # Log the output data information
        if self.config.mmd_out is not None:
            self.logger.info(f'Writing markdown files to {self.config.mmd_out}')
        else:
            self.logger.info('`mmd_out` not specified, will not write markdown files.')

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:  # noqa: PLR0912, PLR0915
        """Parse a PDF file and extract markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.

        Returns
        -------
        list[dict[str, Any]]
            The extracted documents.
        """
        import torch
        from nougat.postprocessing import markdown_compatible
        from nougat.utils.dataset import LazyDataset
        from torch.utils.data import ConcatDataset
        from torch.utils.data import DataLoader

        pdfs = [Path(pdf_file) for pdf_file in pdf_files]

        # HOUSEKEEPING
        # from adaparse.parsers.device_utils import resolve_device # legacy

        device_str = resolve_device()
        device = torch.device(device_str)

        if self.config.batchsize <= 0:
            self.config.batchsize = 1

        # extract formatting arguments
        align_long_axis = bool(self.model.encoder.align_long_axis)
        input_size = list(self.model.encoder.input_size)
        random_padding = False
        # - combine
        prepared_arg_triplet = (align_long_axis, input_size, random_padding)

        datasets = []
        for pdf in pdfs:
            if not pdf.exists():
                self.logger.warning(f'Could not find {pdf}. Skipping.')
                continue
            if self.config.mmd_out is not None:
                out_path = self.config.mmd_out / pdf.with_suffix('.mmd').name
                if out_path.exists() and not self.config.recompute:
                    self.logger.info(
                        f'Skipping {pdf.name}: already extracted. '
                        ' Use --recompute config to override extraction.'
                    )
                    continue

            try:
                # dataset
                dataset = LazyDataset(
                    pdf,
                    partial(prepare_input_sc, prep_args=prepared_arg_triplet),
                )

            # PdfStreamError, ValueError, KeyError, pypdf.errors.PdfReadError,
            # and potentially other exceptions can be raised here.
            except Exception:
                self.logger.info(f'Could not load file {pdf!s}.')
                continue
            datasets.append(dataset)

        # If there are no PDFs to process, return None
        if len(datasets) == 0:
            return None

        # dataloader arguments
        dl_kwargs = dict(
            batch_size=self.config.batchsize,
            pin_memory=True,
            num_workers=self.config.num_workers,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )
        if self.config.num_workers > 0:
            dl_kwargs['prefetch_factor'] = self.config.prefetch_factor

        # dataloader
        dataloader = DataLoader(ConcatDataset(datasets), **dl_kwargs)

        documents = []
        predictions = []
        file_index = 0
        page_num = 0
        model_outputs = []

        start = time.time()

        # = = = = = = = = = = = = = = = = = = = = = = =
        #
        # = = = = = = = = = = = = = = = = = = = = = = =
        self.logger.info(f"\nprint(len(dataloader)): {len(dataloader)}")
        # = = = = = = = = = = = = = = = = = = = = = = =

        # First pass to get the model outputs
        for sample, is_last_page in dataloader:
            model_output = self.model.inference(
                image_tensors=sample, early_stopping=self.config.skipping
            )
            model_outputs.append((model_output, is_last_page))

        self.logger.info(
            f'First pass took {time.time() - start:.2f} seconds. '
            'Processing the model outputs.'
        )
        start = time.time()

        # Load PyMuPDF
        # = = = = = = = = = = = = = = = = = = = = = = =
        #self.logger.info(f"\nprint(len(model_outputs)): {len(model_outputs)}")
        # = = = = = = = = = = = = = = = = = = = = = = =

        # Second pass to process the model outputs
        for i, (model_output, is_last_page) in enumerate(model_outputs):
            # check if model output is faulty
            for j, output in enumerate(model_output['predictions']):
                if page_num == 0:
                    self.logger.info(
                        'Processing file %s with %i pages'
                        % (
                            datasets[file_index].name,
                            datasets[file_index].size,
                        )
                    )

                    # parser
                    doc_parser_name = 'nougat'

                    # open file : fill missing pages
                    doc = None
                    if self.config.fill_missing_pages:
                        try:
                            doc = pymupdf.open(datasets[file_index].name)
                        except Exception:
                            self.logger.warning("PyMuPDF open failed; falling back to Nougat only.")
                            doc = None

                page_num += 1
                if output.strip() == '[MISSING_PAGE_POST]':
                    # uncaught repetitions -- potentially empty page
                    if self.config.fill_missing_pages:
                        if page_num < len(doc):
                            filled_page_text = '\n\n' + doc.load_page(page_num).get_text() + '\n\n'
                            doc_parser_name = 'nougat/pymupdf'
                        else:
                            filled_page_text = f'\n\n[MISSING_PAGE_EMPTY_FAILED_FILL:{page_num}]\n\n'
                        self.logger.warning(
                                    f'Fill empty page {page_num} via PyMuPDF as it is claimed EMPTY.'
                            )
                        predictions.append(filled_page_text)
                    else:
                        self.logger.warning(
                                f'Skipping page {page_num} as it is claimed EMPTY.'
                        )
                        predictions.append(f'\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n')
                elif self.config.skipping and model_output['repeats'][j] is not None:
                    if model_output['repeats'][j] > 0:
                        # Likely incomplete and truncated output
                        if self.config.fill_missing_pages:
                            if page_num < len(doc):
                                filled_page_text = '\n\n' + doc.load_page(page_num).get_text() + '\n\n'
                                doc_parser_name = 'nougat/pymupdf'
                            else:
                                filled_page_text = f'\n\n[MISSING_PAGE_FAIL_FAILED_FILL:{page_num}]\n\n'
                            self.logger.warning(
                                f'Filled page {page_num} via PyMuPDF due to repetitions.'
                            )
                            predictions.append(filled_page_text)
                        else:
                            self.logger.warning(
                                f'Skipping page {page_num} due to repetitions.'
                            )
                            predictions.append(f'\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n')
                    else:
                        # Nougat parsing failure ("FAIL")
                        if self.config.fill_missing_pages:
                            if page_num < len(doc):
                                filled_page_text = '\n\n' + doc.load_page(page_num).get_text() + '\n\n'
                                doc_parser_name = 'nougat/pymupdf'
                            else:
                                filled_page_text = f'\n\n[MISSING_PAGE_EMPTY_FAILED_FILL:{page_num}]\n\n'
                            predictions.append(filled_page_text)
                        else:
                            predictions.append(
                                f'\n\n[MISSING_PAGE_EMPTY:'
                                f'{i * self.config.batchsize + j + 1}]\n\n'
                            )

                else:
                    if self.config.markdown:
                        output = markdown_compatible(output)  # noqa: PLW2901
                    predictions.append(output)
                if is_last_page[j]:
                    out, page_indices = build_doc_and_indices(predictions)

                    # close file : fill missing pages
                    if self.config.fill_missing_pages and doc is not None:
                        doc.close()

                    # metadata
                    metadata = {'page_char_idx': page_indices}

                    # write document
                    document = {
                        'path': str(is_last_page[j]),
                        'text': out,
                        'metadata': metadata,
                        'parser': doc_parser_name,
                    }
                    documents.append(document)

                    if self.config.mmd_out is not None:
                        # writing the outputs to the markdown files a separate
                        # directory.
                        out_path = (
                            self.config.mmd_out
                            / Path(is_last_page[j]).with_suffix('.mmd').name
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(out, encoding='utf-8')

                    predictions = []
                    page_num = 0
                    file_index += 1

        self.logger.info(
            f'Second pass took {time.time() - start:.2f} seconds. '
            'Finished processing the model outputs.'
        )

        # workflow return
        return documents
