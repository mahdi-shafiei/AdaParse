"""The Nougat PDF parser."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal, List, Dict
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
    recompute: bool = False
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
            # LEGACY
            #from nougat.utils.checkpoint import get_checkpoint

            print(
                f'Checkpoint not found in {value}.'
                'Rebuild AdaParse via `source ./scripts/initial_setup.sh` or '
                'download weights directly by running'
                '`source ./scripts/weights/download_nougat_checkpoint.sh`'
            )
            # LEGACY
            # value = get_checkpoint(value, model_tag='0.1.0-base')
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

        # set config
        self.config = config

        # DEBUG
        ##print('\n\nself.config')
        #print(self.config)

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

        # assign
        self.model = model
        self.processor = processor

        # device/dtype
        self.device = next(self.model.parameters()).device
        self.dtype  = next(self.model.parameters()).dtype

        # DEBUG
        self.logger.info(f'self.device: {self.device}')

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

        # dataset
        datasets: List[LazyDataset] = []
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
            except:
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
        # - log
        self.logger.info(f"\nprint(len(dataloader)): {len(dataloader)}")

        documents: List[Dict[str, Any]] = []
        predictions: List[str] = []
        file_index = 0
        page_num = 0
        model_doc_outputs = []

        # - - - - - - - - - - - - - - - -
        # 1st pass: pure Nougat inference
        # - - - - - - - - - - - - - - - -
        img_ = 0
        print(f"Time: {time.time()}\nBefore amp_infer_context()")
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
                # DEBUG
                print()
                print(f'decoder_output : {decoder_output}')
                print()

                # post-processing
                processed_doc_output = process_decoder_output(decoder_output=decoder_output,
                                                              tokenizer=self.processor.tokenizer)

                # DEBUG
                print()
                print(f'processed_doc_output : {processed_doc_output}')
                print()

                # append
                model_doc_outputs.append((processed_doc_output, is_last_page))

        # appendls
        end = time.time()
        # - log
        self.logger.info(
            f'[TIME] 1st pass (Nougat Inference): {end - start:.2f}[s]\nFor {img_} images'
        )

        # - - - - - - - - - - - - - - - -
        # 2nd pass: (optional) fill-in
        # - - - - - - - - - - - - - - - -
        start = time.time()

        # document-loop
        for _, (doc_output, is_last_page) in enumerate(model_doc_outputs):
            # fill-in model output
            page_num = 0
            doc = None

            # page loop
            for j, page_output in enumerate(doc_output['predictions']):
                # first page only
                if page_num == 0:
                    predictions=[]
                    doc_parser_name = 'nougat'

                    # open file : fill missing pages
                    if self.config.fill_missing_pages and doc is None:
                        doc = safe_doc_open(datasets[file_index].name, self.logger)

                # every page
                # - detect Nougat failure
                if (('MISSING_PAGE' in page_output) or (len(page_output) < 20)) and (doc is not None):
                    prev_len = len(page_output)
                    page_output_tmp = '\n\n' + doc.load_page(page_num).get_text() + '\n\n'
                    doc_parser_name = 'nougat/pymupdf'
                    new_len = len(page_output_tmp)
                    if new_len > prev_len:
                        page_output = page_output_tmp
                # - formatting (source-independent)
                if self.config.markdown:
                    prev_len = len(page_output)
                    page_output = markdown_compatible(page_output)
                    new_len = len(page_output)
                    # - -

                # append
                predictions.append(page_output)

                # last page only
                if is_last_page[j]:
                    # - -
                    out, page_indices = build_doc_and_indices(predictions)
                    # - close doc
                    if self.config.fill_missing_pages and (doc is not None):
                        safe_doc_close(doc, self.logger)
                        doc = None

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

                    # write explicit .mmd
                    if self.config.mmd_out is not None:
                        # - -
                        out_path = (
                            self.config.mmd_out
                            / Path(is_last_page[j]).with_suffix('.mmd').name
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(out, encoding='utf-8')

                    # reset
                    predictions = []
                    page_num = 0
                    file_index += 1
                else:
                    # - increment page
                    page_num+=1

        end = time.time()
        self.logger.info(
            f'[TIME] 2nd pass (Fill-in): {end - start:.2f}[s].'
        )

        # workflow return
        return documents
