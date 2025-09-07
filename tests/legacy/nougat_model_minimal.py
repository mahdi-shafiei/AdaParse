from __future__ import annotations

import os

# torch on aurora
import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex

from pathlib import Path
from typing import List
import numpy as np
import argparse
from functools import partial

from transformers import VisionEncoderDecoderModel, NougatProcessor
from transformers import StoppingCriteriaList

from adaparse.parsers.nougat_parser.decoding import process_decoder_output
from adaparse.parsers.nougat_parser.decoding import StoppingCriteriaScores
from adaparse.device_utils import move_to_custom_device, amp_infer_context
from adaparse.parsers.nougat_inference_utils import prepare_input_sc
#from adaparse.parsers.nougat_ import NougatParserConfig
from adaparse.convert import WorkflowConfig

from adaparse.parsers.nougat_parser.utils.dataset import LazyDataset
from torch.utils.data import ConcatDataset, DataLoader

from adaparse.parsers.nougat_parser.utils.eval import load_mmd, page_similarity

def main(args):
    # load config
    config_path = Path(args.config_path)
    assert config_path.is_file(), f"Path to config invalid. Not a file: {config_path}"

    # load config
    config = WorkflowConfig.from_yaml(config_path)
    # - overwrite `pdf_dir` from 20 test PDFs to single 20-page test PDF.
    config.pdf_dir = args.pdf_dir

    # load model/processor
    processor = NougatProcessor.from_pretrained(config.parser_settings.checkpoint,
                                                use_fast=True,
                                                local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(config.parser_settings.checkpoint,
                                                      local_files_only=True)

    # GROUNDTRUTH
    assert Path(args.gt_mmd).is_file(), f"Path to groundtruth MMD is not file: {args.gt_mmd}"
    ground_truth_pages = load_mmd(args.gt_mmd)

    # MODEL
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id

    # device management: ./data/nougat_*.mmd was generated under full precision
    model = move_to_custom_device(model, bf16=False)

    # DEBUG
    if 'ipex' in globals():
        model = ipex.optimize(model, dtype=model.dtype)

    # 1. destination
    pdf_dir = Path(config.pdf_dir)
    out_dir = Path(config.out_dir)
    # - create
    if not pdf_dir.is_dir():
        pdf_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    # PDFs
    assert pdf_dir.is_dir(), "PDF directory does not exist"
    pdf_files = [pdf_dir / f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f'PDF Files: {pdf_files}')

    assert len(pdf_files) > 0, "No PDFs present in `pdf_dir`"
    pdfs = [Path(pdf_file) for pdf_file in pdf_files]

    # 2. Nougat constants
    align_long_axis = False
    input_size = [896, 672]
    random_padding = False
    # - combine
    prepared_arg_triplet = (align_long_axis, input_size, random_padding)

    # DATASET
    datasets: List[LazyDataset] = []
    for pdf in pdfs:
        if not pdf.exists():
            #self.logger.warning(f'Could not find {pdf}. Skipping.')
            print(f'Could not find {pdf}. Skipping.') # help
            continue

        # dataset
        try:
            dataset = LazyDataset(
                pdf,
                partial(prepare_input_sc, prep_args=prepared_arg_triplet),
            )
        except Exception:
            #self.logger.info(f'Could not load file {pdf!s}.')
            print(f'Could not load file {pdf!s}.')
            continue
        datasets.append(dataset)

    # no data
    assert len(datasets) > 0, "PDFs present in `pdf_dir` but LazyDataset is empty."

    # DATALOADER
    # - dataloader arguments
    dl_kwargs = dict(
        batch_size=10,
        pin_memory=True,
        num_workers=config.parser_settings.num_workers,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )
    # - prefatch
    if config.parser_settings.num_workers > 0:
        dl_kwargs['prefetch_factor'] = config.parser_settings.prefetch_factor

    # dataloader
    dataloader = DataLoader(ConcatDataset(datasets), **dl_kwargs)

    # INFERENCE
    predictions = []
    for sample, _ in dataloader:
        # adaptive context
        with amp_infer_context(model=model):
            # encoder:
            encoded = processor(images=sample, return_tensors="pt").to(device=model.device, dtype=model.dtype)

            # decoder: generate from full model
            decoder_output = model.generate(pixel_values=encoded.pixel_values,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            do_sample=False,
                                            max_new_tokens=1024,
                                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()])
             )

            # post-processing
            output = process_decoder_output(decoder_output=decoder_output, tokenizer=processor.tokenizer)

            # track
            preds = output['predictions']
            predictions.extend(preds)

    #text comparison
    similarities = [page_similarity(pred, gt) for pred, gt in zip(predictions, ground_truth_pages)]

    if np.mean(similarities) > args.accept_rate:
        print()
        print(f"Avg. page text similarity: {np.mean(similarities):.2f}%")
        print('Test passed!')

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract text from PDF images using Nougat')
    parser.add_argument('--config_path', '-c', help='Path to machine-specific Nougat config')
    parser.add_argument('--pdf_dir', default='./data/single_pdf', help='Directory with `test.pdf` to overwrite Nougat config`s `pdf_dir`.')
    parser.add_argument('--gt_mmd', default='./data/groundtruth/nougat_base.mmd', help='Path to groundtruth MMD generated by original Nougat implementation.')
    parser.add_argument('--accept_rate', type=float, default=95.0, help='Similarity threshold (average rapidfuzz str similarity across pages).')
    args = parser.parse_args()

    # entry
    main(args)
