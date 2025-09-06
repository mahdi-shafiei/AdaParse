from __future__ import annotations

# torch on aurora
import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex

import os
from pathlib import Path
from typing import List
import argparse
from functools import partial

from transformers import VisionEncoderDecoderModel, NougatProcessor
from transformers import StoppingCriteriaList

from adaparse.parsers.nougat_parser.decoding import process_decoder_output, StoppingCriteriaScores
from adaparse.device_utils import move_to_custom_device, amp_infer_context
from adaparse.parsers.nougat_inference_utils import prepare_input_sc

from adaparse.parsers.nougat_parser.utils.dataset import LazyDataset
from torch.utils.data import ConcatDataset, DataLoader

def main(args):
    # load model/processor
    checkpoint = '/home/siebenschuh/AdaParse/models/facebook__nougat-base'
    processor = NougatProcessor.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, local_files_only=True)

    # = = = = = = =
    #    MODEL
    # = = = = = = =
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id

    # device management
    model = move_to_custom_device(model, bf16=False)

    # DEBUG
    if 'ipex' in globals():
        model = ipex.optimize(model, dtype=model.dtype)

    # = = = = = =
    #  DATASET
    # = = = = = =
    # 1. destination
    args.pdf_dir = Path(args.pdf_dir)
    args.out_dir = Path(args.out_dir)
    # - create
    if not args.pdf_dir.is_dir():
        args.pdf_dir.mkdir(parents=True, exist_ok=True)
    if not args.out_dir.is_dir():
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load PDFs
    assert args.pdf_dir.is_dir(), "PDF directory does not exist"
    # TODO
    pdf_files = [args.pdf_dir / f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')]

    # DEBUG
    print('pdf_files')
    print(pdf_files)

    assert len(pdf_files) > 0, "No PDFs in `pdf_dir`"
    pdfs = [Path(pdf_file) for pdf_file in pdf_files]

    # 2. Fix arguments
    align_long_axis = False # hack, from SwinEncoder config
    input_size = [896, 672] # hack, from SwinEncoder config
    random_padding = False
    # - combine
    prepared_arg_triplet = (align_long_axis, input_size, random_padding)

    # dataset
    datasets: List[LazyDataset] = []
    for pdf in pdfs:
        if not pdf.exists():
            #self.logger.warning(f'Could not find {pdf}. Skipping.')
            print(f'Could not find {pdf}. Skipping.') # help
            continue

        # TODO: fix dir to dump MMD into
        #if self.config.mmd_out is not None:
        if True:
            out_path = args.out_dir / 'mmd' / pdf.with_suffix('.mmd').name
            if out_path.exists() and not self.config.recompute:
                #logger.info(
                #    f'Skipping {pdf.name}: already extracted. '
                #    ' Use --recompute config to override extraction.'
                #)
                print(f'Skipping {pdf.name}: already extracted. ') # help
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
            #self.logger.info(f'Could not load file {pdf!s}.')
            print(f'Could not load file {pdf!s}.') # help
            continue
        datasets.append(dataset)

    # If there are no PDFs to process, return None
    if len(datasets) == 0:
        return None

    # = = = = = = = =
    #  DATALOADER
    # = = = = = = = =

    # dataloader arguments
    dl_kwargs = dict(
        batch_size=args.batch_size, # self.config.batchsize,
        pin_memory=True,
        num_workers=args.num_workers, # self.config.num_workers,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )
    #if self.config.num_workers > 0:
    if args.num_workers > 0:
        dl_kwargs['prefetch_factor'] = 2 #self.config.prefetch_factor

    # dataloader
    dataloader = DataLoader(ConcatDataset(datasets), **dl_kwargs)

    documents = [] # List[Dict[str, Any]] = []
    predictions = []
    file_index = 0
    page_num = 0
    model_outputs = []

    # = = = = = = = =
    #  INFERENCE
    #   (Nougat)
    # = = = = = = = =
    for sample, is_last_page in dataloader:
        # adaptive context
        with amp_infer_context(model=model):
            # encoder:
            encoded = processor(images=sample, return_tensors="pt").to(device=model.device, dtype=model.dtype)
            pixel_values = encoded.pixel_values

            # decoder: generate from full model
            decoder_output = model.generate(pixel_values=pixel_values,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            do_sample=False,
                                            max_new_tokens=1024,
                                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()])
             )

            #filter
            output = process_decoder_output(decoder_output=decoder_output, tokenizer=processor.tokenizer)

            # print
            preds = output['predictions']

            # DEBUG
            print('\n\n')
            print(preds)

    return

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Extract text from PDF images using Nougat')
    parser.add_argument('pdf_dir', help='Path to the image file (e.g., ./data/)')
    parser.add_argument('--out_dir', '-o', help='Output file to save the extracted text (optional)')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    main(args)
