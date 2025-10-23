"""The Nougat PDF parser."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal, List, Dict, Optional
from pydantic import field_validator
from transformers import StoppingCriteriaList

from adaparse.parsers.base import BaseParser
from adaparse.parsers.base import BaseParserConfig
from adaparse.device_utils import build_doc_and_indices
from adaparse.device_utils import move_to_custom_device
from adaparse.device_utils import amp_infer_context
from adaparse.parsers.nougat_inference_utils import prepare_input_sc

from adaparse.utils import exception_handler
from adaparse.utils import setup_logging

from adaparse.parsers.nougat_parser.decoding import StoppingCriteriaScores
from adaparse.parsers.nougat_parser.decoding import process_decoder_output
from adaparse.parsers.pymupdf_parser.utils import safe_doc_open, safe_doc_close

__all__ = [
    'NougatParser',
    'NougatParserConfig',
]
class NougatParserConfig(BaseParserConfig):
    """Settings for the Nougat parser."""

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
    recompute: bool = True
    # Fill-in of failed/empty pages via PyMuPDF (False implies pure Nougat)
    fill_missing_pages: bool = False
    # Use float32 (False: bfloat32)
    full_precision: bool = False
    # Whether to format the output as markdown.
    markdown: bool = True
    # Skip if the model falls in repetition.
    skipping: bool = True
    # The directory to write the logs to.
    nougat_logs_path: Path
    # Minimum number of chars to be considered likely page
    minimum_chars_per_page: int = 20

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
            print(
                f'Checkpoint not found in {value}.'
                'Rebuild Repo via `source ./scripts/initial_setup.sh` or '
                'download weights directly by running'
                '`source ./scripts/weights/download_nougat_checkpoint.sh`'
            )

        return value


class NougatParser(BaseParser):
    """Warmstart interface for the updated Nougat parser.

    Initialization loads the Nougat weights (ViT encoder, BART decoder) into memory and registers them in a
    global registry unique to the current process. This ensures that the models
    are only loaded once per worker process - hence warmstart.
    """

    def __init__(self, config: NougatParserConfig) -> None:
        """Initialize the Nougat parser."""
        # torch on aurora
        import torch
        if torch.xpu.is_available():
            import intel_extension_for_pytorch as ipex

        from transformers import VisionEncoderDecoderModel
        from transformers import NougatProcessor

        # logger setup
        self.logger = setup_logging('adaparse_nougat', config.nougat_logs_path)

        # set config
        self.config = config

        # load model/processor
        model = VisionEncoderDecoderModel.from_pretrained(self.config.checkpoint,
                                                          local_files_only=True)
        processor = NougatProcessor.from_pretrained(self.config.checkpoint,
                                                    use_fast=True,
                                                    local_files_only=True)

        # set model config
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        model.config.pad_token_id           = processor.tokenizer.pad_token_id
        model.config.eos_token_id           = processor.tokenizer.eos_token_id

        # device management
        model = move_to_custom_device(model, bf16=not(self.config.full_precision))

        # optimization
        use_ipex = hasattr(torch, "xpu") and torch.xpu.is_available()
        if use_ipex:
            model = ipex.optimize(model, dtype=model.dtype)

        # DEBUG
        #self.logger.info(f'model                     : {model}')
        self.logger.info(f'hasattr(torch, `xpu`)     : {hasattr(torch, "xpu")}')
        self.logger.info(f'torch.xpu.is_available()  : {torch.xpu.is_available()}')
        self.logger.info(f'use_ipex                  : {use_ipex}')
        self.logger.info(f'bf16                      : {not(self.config.full_precision)}')

        # assign
        self.model = model
        self.processor = processor

        # device/dtype
        self.device = next(self.model.parameters()).device
        self.dtype  = next(self.model.parameters()).dtype

        # DEBUG
        self.logger.info(f'self.device: {self.device}')

        # Log the output data information
        if self.config.mmd_out is not None:
            self.logger.info(f'Writing markdown files to {self.config.mmd_out}')
        else:
            self.logger.info('`mmd_out` not specified, will not write markdown files.')

    @exception_handler(default_return=None)
    def parse(self,
              pdf_files: list[str],
              pages_lists: Optional[List[Optional[List[int]]]] = None) -> list[dict[str, Any]] | None:
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
        if torch.xpu.is_available():
            import intel_extension_for_pytorch as ipex
        from adaparse.parsers.nougat_parser.utils.dataset import LazyDataset
        from adaparse.parsers.nougat_parser.postprocessing import markdown_compatible

        from torch.utils.data import ConcatDataset
        from torch.utils.data import DataLoader

        pdfs = [Path(pdf_file) for pdf_file in pdf_files]

        # batch size
        if self.config.batchsize <= 0:
            self.config.batchsize = 1

        # Nougat-specific constants
        align_long_axis = False
        input_size = [896, 672]
        random_padding = False

        # - combine
        prepared_arg_triplet = (align_long_axis, input_size, random_padding)

        # document-wise (no pages_lists for subsetting provided)
        if pages_lists is None:
            pages_lists = [None] * len(pdfs)
            pagewise_flag = False
        else:
            pagewise_flag = True
            # check if page lists are consistent
            if len(pages_lists) != len(pdfs):
                raise ValueError("len(pages_lists) must equal len(pdf_files)")
            for i, lst in enumerate(pages_lists):
                if not lst:
                    raise ValueError(f"pages_lists[{i}] must be non-empty")

        # dataset
        datasets: List[LazyDataset] = []
        for pdf, pages in zip(pdfs, pages_lists):

            # DEBUG
            #print(f'\npages: {pages}')

            if not pdf.exists():
                self.logger.warning(f'Could not find {pdf}. Skipping.')
                continue

            # Markdown output
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
                    pdf=pdf,
                    prepare=partial(prepare_input_sc, prep_args=prepared_arg_triplet),
                    pages=pages,
                )
            except:
                self.logger.info(f'Could not load file {pdf!s}.')
                continue
            datasets.append(dataset)

        # DEBUG: status dataset
        #print(f'\n\n= = = = = = = = = = = = = = = = =\nlen(dataset) : {len(datasets)}')

        # If there are no PDFs to process, return None
        if len(datasets) == 0:
            return None

        # dataloader arguments
        # - only shuffle=False allows reconsruction
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

        documents: List[Dict[str, Any]] = []
        predictions: List[str] = []
        model_doc_outputs = []

        # - - - - - - - - - - - - - - - -
        # 1st pass: pure Nougat inference
        # - - - - - - - - - - - - - - - -
        # adaptive context
        with amp_infer_context(model=self.model):
            # compile model
            try:
                self.model = torch.compile(self.model, fullgraph=True)
            except:
                self.logger = setup_logging('[WARNING] Failed to compile model',
                                            self.config.nougat_logs_path)
            start = time.time()
            # inference loop
            for sample, is_last_page in dataloader:
                # encoder:
                encoded = self.processor(images=sample,
                                         return_tensors="pt").to(device=self.device,
                                                                 dtype=self.dtype)

                # decoder: generate from full model
                decoder_output = self.model.generate(pixel_values=encoded.pixel_values,
                                                     return_dict_in_generate=True,
                                                     output_scores=True,
                                                     do_sample=False,
                                                     max_new_tokens=1024,
                                                     bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                                     stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()])
                                                     )

                # post-processing
                processed_doc_output = process_decoder_output(decoder_output=decoder_output,
                                                              tokenizer=self.processor.tokenizer)

                # append
                model_doc_outputs.append((processed_doc_output, is_last_page))

        # time
        end = time.time()
        # - log
        self.logger.info(
            f'[TIME] 1st pass (Nougat Inference): {end - start:.2f}[s]'
        )

        # - - - - - - - - - - - - - - - -
        # 2nd pass: (optional) fill-in
        # - - - - - - - - - - - - - - - -
        # must initialize doc, page_num, file_index here
        file_index = 0
        doc = None
        page_num = 0

        # containers for page-wise output
        pagewise_documents: list[dict[str, dict[int, str]]] = []
        per_file_pages = {}   # map original_page_index -> page_text

        # 1st: loop over batches of documents (not documents themselves)
        start = time.time()
        for _, (doc_output, is_last_page_file_name) in enumerate(model_doc_outputs):
            # DEBUG
            self.logger.info(
                f'datasets[file_index].name: {datasets[file_index].name}'
            )

            # 2nd: page loop
            for j, page_output in enumerate(doc_output['predictions']):
                # first page only
                if page_num == 0:
                    # list of page text `predictions`
                    predictions=[]
                    # page processing attributes
                    missing_pages_indices = []
                    overly_short_page_indices = []
                    filled_in_page_indices = []
                    # parser name
                    doc_parser_name = 'nougat'
                    if pagewise_flag:
                        per_file_pages = {}

                    # open file: fill missing pages
                    if self.config.fill_missing_pages and doc is None:
                        doc = safe_doc_open(datasets[file_index].name, self.logger)

                # (optional) fill_missing_pages handling
                # - assess
                page_has_missing_substring = 'MISSING_PAGE' in page_output
                page_text_too_short = len(page_output) < self.config.minimum_chars_per_page
                # - infer goodness of page text
                page_likely_subpar = page_has_missing_substring or page_text_too_short
                # - track reason
                if self.config.fill_missing_pages and page_likely_subpar and (doc is not None):
                    # open page text via extractor
                    if pagewise_flag:
                        original_idx = pages_lists[file_index][page_num]
                        fallback_text = '\n\n' + doc.load_page(original_idx).get_text() + '\n\n'
                    else:
                        fallback_text = '\n\n' + doc.load_page(page_num).get_text() + '\n\n'
                    new_len = len(fallback_text)

                    # heuristic: assign for more tokens
                    if new_len > len(page_output):
                        page_output = fallback_text
                        doc_parser_name = 'pymupdf'
                        filled_in_page_indices.append(page_num)

                    # track reason
                    if page_has_missing_substring:
                        missing_pages_indices.append(page_num)
                    else:
                        overly_short_page_indices.append(page_text_too_short)

                # - formatting (source-independent)
                if self.config.markdown:
                    page_output = markdown_compatible(page_output)

                # ===== Divergence: by_doc vs by_page(subset) assembly =====
                if not pagewise_flag:
                    # by_doc : collect strings to later join
                    predictions.append(page_output)
                else:
                    # subset mode: store under ORIGINAL page index
                    original_idx = pages_lists[file_index][page_num]  # e.g., 0,4,6
                    per_file_pages[original_idx] = page_output

                # last page only / last page boundary
                if is_last_page_file_name[j]:
                    # by_doc
                    if not pagewise_flag:
                        # process
                        joined_text, page_indices = build_doc_and_indices(predictions)
                        # - close doc
                        #if self.config.fill_missing_pages and (doc is not None):
                        if doc is not None:
                            safe_doc_close(doc, self.logger)
                            doc = None

                        # metadata
                        metadata = {
                            'page_char_idx': page_indices,
                            'page_with_missing_substring_idx' : missing_pages_indices,
                            'page_that_are_short_idx' : overly_short_page_indices,
                            'filled_in_page_idx' : filled_in_page_indices,
                        }

                        # write document
                        document = {
                            'path': str(is_last_page_file_name[j]),
                            'text': joined_text,
                            'metadata': metadata,
                            'parser': doc_parser_name,
                        }
                        documents.append(document)

                        # write explicit .mmd in `by_doc` mode only
                        if self.config.mmd_out is not None:
                            # - -
                            out_path = (
                                self.config.mmd_out
                                / Path(is_last_page_file_name[j]).with_suffix('.mmd').name
                            )
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_text(joined_text, encoding='utf-8')

                    # by_page: 1 dict per file
                    else:
                        # page-wise assembly: one dict per file
                        ordered = dict(sorted(per_file_pages.items(), key=lambda kv: kv[0]))
                        pagewise_documents.append({str(is_last_page_file_name[j]): ordered})

                        # close if open (not used in subset mode, but harmless)
                        if self.config.fill_missing_pages and (doc is not None):
                            safe_doc_close(doc, self.logger)
                            doc = None
                        per_file_pages = None

                    # reset
                    file_index += 1
                    # reset page
                    page_num = 0
                else:
                    # increment
                    page_num += 1

            # - batch completed
        # - dataset finished
        # - loop done
        end = time.time()
        self.logger.info(
            f'[TIME] 2nd pass (Fill-in): {end - start:.2f}[s].'
        )

        # Return format:
        # - by_doc : List[Dict[str, Any]]
        # - by_page : List[Dict[str, Dict[int, str]]]
        if not pagewise_flag:
            return documents
        else:
            return pagewise_documents
