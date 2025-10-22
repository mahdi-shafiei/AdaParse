"""The AdaParse PDF parser."""

from __future__ import annotations

import functools
from abc import ABC
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional, Sequence
from itertools import chain, islice

import random
import numpy as np
import torch
from enum import Enum
from pydantic import BaseModel
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from adaparse.parsers.base import BaseParser
from adaparse.parsers.nougat import NougatParser
from adaparse.parsers.nougat import NougatParserConfig
from adaparse.parsers.pymupdf import PyMuPDFParser
from adaparse.parsers.pymupdf import PyMuPDFParserConfig
from adaparse.timer import Timer
from adaparse.utils import exception_handler
from adaparse.device_utils import move_to_device_accelerator

__all__ = [
    'AdaParse',
    'AdaParseConfig',
]
class PredMode(str, Enum):
    """Mode of prediction (regression) and hence delegation of AdaParse: by page or doc"""
    by_doc = "by_doc"
    by_page = "by_page"

class TextDataset(Dataset):
    """Dataset for sequence regression via cls."""

    def __init__(self, texts: list[str]) -> None:
        """Initialize the dataset."""
        self.texts = texts

    def __len__(self) -> int:
        """Return the number of texts."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        """Return a sequence."""
        return self.texts[idx]

class TextRegressionConfig(BaseModel):
    """Settings for the text quality regression."""

    alpha: float = Field(
        description='Max. proportion of high-quality parser.',
    )
    prediction_mode: PredMode = Field(
        default=PredMode.by_doc,
        description='Mode/granularity with which prediction/delegation occurs.',
    )
    weights_path: Path | None = Field(
        default=None,
        description='The path to the fine-tuned model weights - overwrites pred granularity.',

    )
    batch_size: int = Field(
        default=8,
        description='The batch size for the regression model.',
    )
    max_character_length: int = Field(
        default=3200,
        description='The maximum length of the input text (in characters).',
    )
    num_data_workers: int = Field(
        default=1,
        description='The number of data workers for the regression.',
    )
    pin_memory: bool = Field(
        default=True,
        description='Whether to pin memory for the regression model.',
    )


class TextRegression(ABC):
    """Text Quality Regression Model."""

    def __init__(self, config: TextRegressionConfig) -> None:
        """Initialize the regression model via CLS."""
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        # prediction model
        if config.prediction_mode == PredMode.by_doc:
            regr_model_name = '7shoe/adaparse-scibert-uncased' # 7shoe/adaparse-specter-docwise
        elif config.prediction_mode == PredMode.by_page:
            regr_model_name = '7shoe/adaparse-scibert-uncased' # 7shoe/adaparse-specter-pagewise
        else:
            raise ValueError("Prediction mode is either `by_doc` or `by_page`")

        # weights path set
        if config.weights_path is not None:
            regr_model_name = config.weights_path
            print(f"weights_path set: Overwrite `prediction_mode` and use model: {regr_model_name}")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            regr_model_name
        )

        # Load the base model
        model = AutoModelForSequenceClassification.from_pretrained(
            regr_model_name, num_labels=6
        )

        # model to device
        model, device = move_to_device_accelerator(model)

        # Set the model to evaluation mode
        model.eval()

        self.config = config
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def decision_function(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        """Return the decision function.

        Parameters
        ----------
        logits : torch.Tensor
            The model logits.

        Returns
        -------
        torch.Tensor
            The decision function result (tensor of ints).
        """
        ...

    @torch.no_grad()
    def predict(self,
                text: list[str]) -> list[int]:
        """Quality regression on the input text.

        Parameters
        ----------
        text : list[str]
            The input text to regress. Either List of full document texts (`by_doc`) or (flattened) page texts (`by_page`)

        Returns
        -------
        list[int]
            The predicted scores.
        """

        # prediction is agnostic to page/doc-mode
        _text = [t[: self.config.max_character_length] for t in text]

        # Create the dataset
        dataset = TextDataset(_text)

        # Create the data collator (tokenization function)
        collater_fn = functools.partial(
            self.tokenizer,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
            return_special_tokens_mask=False,
        )

        # Create the data loader
        dataloader = DataLoader(
            dataset,
            collate_fn=collater_fn,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_data_workers,
        )

        # Collect the predictions
        predictions = []

        # Iterate over each batch of the data loader
        for batch in dataloader:
            # Move the inputs to the appropriate device
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            # Run the model forward pass
            outputs = self.model(**inputs)

            # Call the decision function
            y_pred = self.decision_function(outputs.logits, self.config.alpha)

            # Collect the predictions
            predictions.extend(y_pred.tolist())

        return predictions


class NougatTextRegression(TextRegression):
    """Text regression for the Nougat parser."""

    def decision_function(
        self,
        logits: torch.Tensor,
        alpha: float,
        disallow_secondary_parsers: bool = True,
        high_quality_parser: str = 'nougat',
        throughput_parser: str = 'pymupdf',
    ) -> np.ndarray:
        """
        Turns the output of a regression model (uni-/multivariate) into that of a classification model.

        Parameters
        ----------
        logits : torch.Tensor
            The model logits.
        alpha : float
            Threshold for selecting the high-quality parser based on its proportion in predictions.
        disallow_secondary_parsers : bool, optional
            If True, restrict predictions to only the high-quality and throughput parsers. Defaults to True.
        high_quality_parser : str, optional
            Name of the high-quality parser. Defaults to 'nougat'.
        throughput_parser : str, optional
            Name of the throughput parser. Defaults to 'pymupdf'.

        Returns
        -------
        np.ndarray
            The predicted classes.
        """
        # default (high-throughput) parser
        parser = 'pymupdf'

        # parser ID map (same as model config)
        # - later : q!=0 as indicator to re-parse
        parser_ids = {
            'pymupdf': 0,
            'nougat': 1,
            'marker': 2,
            'pypdf': 3,
            'grobid': 4,
            'tesseract': 5,
        }

        # validate parser_ids
        required_keys = {parser, high_quality_parser, throughput_parser}
        missing_keys = required_keys - parser_ids.keys()
        if missing_keys:
            raise ValueError(
                f'Missing required parsers in parser_ids: {missing_keys}'
            )

        # detach/convert convert logits to NumPy array
        logits_np = logits.cpu().numpy()

        # Multivariate case: Take the argmax along the last dimension
        pred_classes = np.argmax(logits_np, axis=-1)

        # Alpha-based adjustments
        alpha_exceed_flag = False
        if 0 < alpha < 1:
            class_counts = Counter(pred_classes)
            alpha_exceed_flag = (
                1.0 * class_counts[parser_ids[high_quality_parser]]
            ) / len(pred_classes) > alpha

        if alpha_exceed_flag:
            hq_scores = logits_np[:, parser_ids[high_quality_parser]]
            top_alpha = int(len(hq_scores) * alpha)
            hq_top_idx = np.argsort(-hq_scores)[:top_alpha]

            logits_2nd_best = np.array(logits_np)
            logits_2nd_best[:, parser_ids[high_quality_parser]] = -np.inf
            censored_pred_classes = logits_2nd_best.argmax(axis=-1)

            if disallow_secondary_parsers:
                censored_pred_classes = np.full(
                    len(censored_pred_classes), parser_ids[throughput_parser]
                )

            censored_pred_classes[hq_top_idx] = parser_ids[high_quality_parser]
            pred_classes = censored_pred_classes

        # Disallow secondary parsers
        if disallow_secondary_parsers:
            valid_ids = {
                parser_ids[high_quality_parser],
                parser_ids[throughput_parser],
            }
            pred_classes = [
                int(pred_i)
                if pred_i in valid_ids
                else parser_ids[throughput_parser]
                for pred_i in pred_classes
            ]

        return np.array(pred_classes)

class AdaParseConfig(
    PyMuPDFParserConfig, NougatParserConfig, TextRegressionConfig
):
    """Settings for the AdaParse parser."""

    # The name of the parser.
    name: Literal['adaparse'] = 'adaparse'  # type: ignore[assignment]

    # Maximum proportion of Nougat parses for the job (performance parameter)
    alpha: float = 0.05

    # Granularity of accuracy prediction (and hence delegation: by page or doc)
    prediction_mode: PredMode = PredMode.by_doc

    # convenience properties to access the configs
    @property
    def pymupdf_config(self) -> PyMuPDFParserConfig:
        """Return the PyMuPDF parser configuration."""
        return PyMuPDFParserConfig()

    @property
    def nougat_config(self) -> NougatParserConfig:
        """Return the Nougat parser configuration."""
        return NougatParserConfig(
            batchsize=self.batchsize,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            checkpoint=self.checkpoint,
            mmd_out=self.mmd_out,
            recompute=self.recompute,
            fill_missing_pages=self.fill_missing_pages,
            full_precision=self.full_precision,
            markdown=self.markdown,
            skipping=self.skipping,
            nougat_logs_path=self.nougat_logs_path,
        )

    @property
    def regression_config(self) -> TextRegressionConfig:
        """Return the text regression model configuration."""
        return TextRegressionConfig(
            alpha=self.alpha,
            prediction_mode=self.prediction_mode,
            weights_path=self.weights_path,
            batch_size=self.batch_size,
            max_character_length=self.max_character_length,
            num_data_workers=self.num_data_workers,
            pin_memory=self.pin_memory,
        )


class AdaParse(BaseParser):
    """Interface for the AdaParse PDF parser."""

    def __init__(self, config: AdaParseConfig) -> None:
        """Initialize the parser."""
        # init PyMuPDF (HT) and Nougat (HQ) parsers
        self.pymupdf_parser = PyMuPDFParser(config=config.pymupdf_config)
        self.nougat_parser = NougatParser(config=config.nougat_config)

        # init the quality check classifier: 0 (PyMuPDF output probably of high quality), 1 (require reparse)
        self.prediction_mode = config.prediction_mode
        self.classifier = NougatTextRegression(config=config.regression_config)


    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""
        # first, parse the PDFs using PyMuPDF
        with Timer('adaparse-pymupdf-parsing', self.unique_id):
            documents = self.pymupdf_parser.parse(pdf_files) # by_doc: list[str]; by_page: list[list[str]]

        # If no documents, there was an error parsing the PDFs with PyMuPDF
        if documents is None:
            return None

        #
        print(f'len(documents) : {len(documents)}')

        # shallow copy
        docs_before_filter = list(documents)

        # quality check regressor
        with Timer('adaparse-quality-check', self.unique_id):
            # parse by prediction mode
            if self.prediction_mode==PredMode.by_doc:
                document_text = [doc['text'] for doc in docs_before_filter]
            elif self.prediction_mode==PredMode.by_page:
                document_page_texts = [doc['metadata']['page_text_list'] for doc in docs_before_filter]
            else:
                raise ValueError("prediction_mode only `by_doc' or `by_page`")

            # qualitites: list[str] or list[list[str]]
            if self.prediction_mode==PredMode.by_doc:
                qualities = self.classifier.predict(document_text)
                # DEBUG
                qualities = [random.choice([0,1]) for _ in range(len(qualities))]
            else:
                # TODO: flatten out for efficiency, recombine?

                # LEGACY (one line)
                #qualities_list = [self.classifier.predict(page_texts) for page_texts in document_page_texts]

                # New:
                # record original lengths (empties ok)
                lengths = [len(page_texts) for page_texts in document_page_texts]
                # flatten input to SciBERT/Specter
                flat_document_texts= list(chain.from_iterable(document_page_texts))
                flat_qualities = self.classifier.predict(flat_document_texts)

                # DBEUG: FAKE QUALITY SIGNALS
                flat_qualities = [random.choice([0,1]) for _ in range(len(flat_qualities))]

                # sanity check
                if len(flat_qualities) != len(flat_document_texts):
                    raise ValueError(f"Prediction length mismatch: got {len(flat_qualities)} for {len(flat_document_texts)} inputs")

                # regroup into flat qualities into qualities_list:list[list[int]]
                it = iter(flat_qualities)
                qualities_list = [list(islice(it, n)) for n in lengths]

            # DEBUG
            if self.prediction_mode == PredMode.by_doc:
                print('len(qualities) : ', len(qualities))
                print('[DOC-MODE] qualities : {qualities}')
            else:
                print('lens(qualities_list) : ', [len(q) for q in qualities_list])
                print('[PAGE-MODE] qualities_list : {qualities_list}')

        # log the percentage of low-quality documents
        if self.prediction_mode==PredMode.by_doc:
            low_quality_num = sum(q != 0 for q in qualities)
            low_quality_percentage = (low_quality_num / max(1, len(qualities))) * 100.0
        else:
            low_quality_num = 0.0
            qualities_total_len = 0
            for qs in qualities_list:
                low_quality_num += sum(q != 0 for q in qs)
                qualities_total_len += len(qs)
            low_quality_percentage = (low_quality_num / max(1, qualities_total_len)) * 100.0

        # status
        print(f'Low-quality documents: {low_quality_percentage:.2f}%')

        # ensures alignment
        if self.prediction_mode==PredMode.by_doc:
            low_quality_pdf_paths = [d["path"] for d, q in zip(docs_before_filter, qualities) if q != 0]
            flawless_documents = [d for d, q in zip(docs_before_filter, qualities) if q == 0]
        else:
            # pagewise
            flawless_documents, repair_documents = [], [] # partition of doc dicts
            low_quality_pdf_paths = []  # PDF paths
            all_low_quality_pages = []  # list[list[int]]
            all_high_quality_pages = [] # list[list[int]]
            # loop docs
            for doc, qualities in zip(docs_before_filter, qualities_list):
                # every single page = good
                if sum(qualities) == 0:
                    flawless_documents.append(doc)
                # at least one bad page
                else:
                    low_quality_pages, high_quality_pages = [], [] # list[int]
                    low_quality_pdf_paths.append(doc["path"])
                    # - loop pages: partition them
                    for page_idx, q in enumerate(qualities):
                        if q != 0:
                            low_quality_pages.append(page_idx)
                        else:
                            high_quality_pages.append(page_idx)
                    # append
                    all_low_quality_pages.append(low_quality_pages)
                    all_high_quality_pages.append(high_quality_pages)
                    repair_documents.append(doc)

        # no document to repair (re-parse)
        if len(low_quality_pdf_paths) == 0:
            return flawless_documents

        # parse the low-quality documents using the Nougat parser
        with Timer('adaparse-nougat-parsing', self.unique_id):
            if self.prediction_mode==PredMode.by_doc:
                nougat_documents = self.nougat_parser.parse(low_quality_pdf_paths)
            else:
                # List[Path], List[List[int]] (file path and to-be repaired PDFs)
                pagewise_documents = self.nougat_parser.parse(low_quality_pdf_paths, all_low_quality_pages)

                # non-empty output
                if pagewise_documents is None:
                    # pagewise_documents: List[{path: {page_idx: text}}]  ->  map[str, Dict[int, str]]
                    page_map = {next(iter(d)): next(iter(d.values())) for d in pagewise_documents}

                    # recombine
                    repaired_documents = []
                    for rep_doc in repair_documents:
                        pdf_file_path = rep_doc['path']
                        page_text_list = list(rep_doc['metadata']['page_text_list'])
                        replacement = page_map.get(pdf_file_path, {})
                        for page_idx, new_text in replacement.items():
                            if 0 <= page_idx < len(page_text_list):
                                page_text_list[page_idx] = str(new_text)
                        # update document dic
                        # - full text of doc
                        rep_doc['text'] = "".join(page_text_list)
                        # - page texts
                        rep_doc['metadata']['page_text_list'] = page_text_list
                        # - recomputes char indices of page breaks
                        page_text_lens = [len(page_t) for page_t in page_text_list]
                        rep_doc['metadata']['page_char_idx'] = np.concatenate([np.array([0]),
                                                                            np.cumsum(page_text_lens[:-1])]).tolist()

                        # append
                        repaired_documents.append(rep_doc)

                    # repaired (partially Nougat-parsed)
                    nougat_documents = list(repaired_documents)
                else:
                    # Nougat returned nothing pagewise; warn and return empty (or fall back, if you prefer)
                    print("Nougat pagewise parse returned no results; producing empty nougat_documents.")
                    # raise warning that nougat went empty
                    nougat_documents = []

        # If Nougat documents were parsed, add them to the output
        if nougat_documents is not None:
            print(f'Nougat parsed documents: {len(nougat_documents)}')
            return flawless_documents + nougat_documents

        # Finally, return the parsed documents from both parsers
        return flawless_documents
