"""PDF conversion workflow."""

from __future__ import annotations

import functools
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor

from adaparse.parsers import ParserConfigTypes
from adaparse.parsl import ComputeSettingsTypes
from adaparse.utils import BaseModel
from adaparse.utils import batch_data
from adaparse.utils import setup_logging

from functools import lru_cache # FOXTROT

# FOXTROT: supress albumentations version check
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1") # crucial
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# X X X X X X X X X X X X X



def parse_pdfs(
    pdf_paths: list[str], output_dir: Path, parser_kwargs: dict[str, Any]
) -> None:
    """Parse a batch of PDF files and write the output to a JSON lines file.

    Parameters
    ----------
    pdf_path : list[str]
        Paths to a batch of PDF file to convert.
    output_dir: Path
        Directory to write the output JSON lines file to.
    parser_kwargs : dict[str, Any]
        Keyword arguments to pass to the parser. Contains an extra `name`
        argument to specify the parser to use.
    """
    import json
    import uuid

    from adaparse.parsers import get_parser # back in
    from adaparse.timer import Timer
    from adaparse.utils import setup_logging

    # Setup logging
    logger = setup_logging('adaparse')

    # Unique ID for logging
    unique_id = str(uuid.uuid4())

    # Initialize the parser. This loads the models into memory and registers
    # them in a global registry unique to the current parsl worker process.
    # This ensures that the models are only loaded once per worker process
    # (i.e., we warmstart the models)

    # FOXTROT X X X X X X X X X X X X X X X X X X
    with Timer('initialize-parser', unique_id):
        parser = get_parser(parser_kwargs, register=True)

    # Process the PDF files in bulk
    #with Timer('parser-parse', unique_id):
    #    documents = parser.parse(pdf_paths)

    # FOXTROT X X X X X X X X X X X X X X X
    from functools import lru_cache

    def _as_torch_dtype(s: str):
        import torch
        s = (s or "bfloat16").lower()
        return {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16,  "float16": torch.float16,
            "fp32": torch.float32,  "float32": torch.float32,
        }.get(s, torch.bfloat16)

    @lru_cache(maxsize=1)
    def _load_hf_nougat(model_src: str, dtype_str: str = "bfloat16", device: str | None = None):
        """Load processor+model once per worker process (parsl safe)."""
        import torch
        from transformers import VisionEncoderDecoderModel, NougatProcessor

        torch_dtype = _as_torch_dtype(dtype_str)
        processor = NougatProcessor.from_pretrained(model_src, use_fast=True, local_files_only=True)
        model = VisionEncoderDecoderModel.from_pretrained(model_src, torch_dtype=torch_dtype, local_files_only=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        return processor, model, device, torch_dtype

    def _pdf_to_pil(
        pdf_path: str,
        dpi: int = 192*5,
        crop_margins: bool = True,
        crop_threshold: int = 200,       # matches legacy logic
    ):
        """
        Render all pages of a PDF to RGB PIL Images.
        - Rasterize (pypdfium2 -> pdf2image fallback)
        - (Optional) crop white margins (replicates legacy prepare_input's crop_margin)
        - DO NOT rotate/resize/pad here; the NougatProcessor handles that.

        Returns: List[PIL.Image.Image] in RGB
        """
        from PIL import Image
        import numpy as np

        def _crop_margin_pil(img: Image.Image) -> Image.Image:
            # Replicates the legacy crop_margin using thresholding + bounding rect.
            # Safe fallbacks to return the original image when no content is found.
            try:
                gray = np.array(img.convert("L"), dtype=np.uint8)
                if gray.size == 0:
                    return img
                # Normalize to [0,255] (same spirit as original)
                gmin, gmax = gray.min(), gray.max()
                if gmax == gmin:
                    return img
                norm = ((gray - gmin) * (255.0 / (gmax - gmin))).astype(np.uint8)
                mask = (norm < crop_threshold).astype(np.uint8) * 255  # text = black-ish
                # Find bounding box of non-white
                ys, xs = np.where(mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    return img
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                # +1 because PIL crop is exclusive on the right/bottom
                return img.crop((x0, y0, x1 + 1, y1 + 1))
            except Exception:
                return img

        images = []

        # --- Try pypdfium2 first ---
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_path)
            scale = dpi / 72.0
            for i in range(len(pdf)):
                page = pdf[i]
                # Apply PDF’s own rotation; render with annotations and AA
                # pypdfium2>=4: render returns a Bitmap. Use flags for quality.
                bitmap = page.render(
                    scale=scale,
                    rotation=page.get_rotation() if hasattr(page, "get_rotation") else 0,
                    # include annotations, text antialiasing, grayscale AA, LCD text AA (when available)
                    flags=pdfium.RenderFlags.ANNOT | pdfium.RenderFlags.OptimizeText,
                )
                pil = bitmap.to_pil()  # already RGB if no alpha
                img = pil.convert("RGB")

                # Some PDFs render with an alpha channel; flatten on white if needed
                if pil.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", pil.size, (255, 255, 255))
                    bg.paste(pil, mask=pil.split()[-1])
                    img = bg

                if crop_margins:
                    img = _crop_margin_pil(img)

                images.append(img)
            return images

        except Exception:
            pass  # fall back

        # --- Fallback: pdf2image ---
        try:
            from pdf2image import convert_from_path
            # convert_from_path returns PIL images; we convert to RGB and crop if desired
            out = []
            for im in convert_from_path(pdf_path, dpi=dpi):
                img = im.convert("RGB")
                if crop_margins:
                    img = _crop_margin_pil(img)
                out.append(img)
            return out

        except Exception as e:
            raise RuntimeError(f"Failed to rasterize PDF '{pdf_path}': {e}")


    def _nougat_infer_pages(pdf_path: str,
                            processor,
                            model,
                            device: str,
                            torch_dtype,
                            batch_pages: int = 6,
                            dpi: int = 192,
                            gen_kwargs: dict | None = None) -> list[str]:
        import torch
        images = _pdf_to_pil(pdf_path, dpi=dpi)
        if not images:
            return []

        gen_kwargs = dict(gen_kwargs or {})
        gen_kwargs.setdefault("num_beams", 1)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("pad_token_id", processor.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", processor.tokenizer.eos_token_id)
        gen_kwargs.setdefault("bad_words_ids", [[processor.tokenizer.unk_token_id]])
        # You can also pass max_new_tokens via YAML -> parser_settings.gen_kwargs

        out_texts: list[str] = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_pages):
                batch = images[i:i+batch_pages]
                # Let the processor handle resize/normalization/padding
                inputs = processor(batch, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype, non_blocking=False)
                out_ids = model.generate(pixel_values, **gen_kwargs)
                texts = processor.batch_decode(out_ids, skip_special_tokens=True)
                texts = [processor.post_process_generation(t, fix_markdown=False) for t in texts]
                out_texts.extend(texts)
        return out_texts
    # X X X X X X X X X X X X X X X X X X X X

    # Pull HF settings from YAML (with sane defaults)
    #model_id         = parser_kwargs.get("model_id", "facebook/nougat-base")
    local_model_dir  = parser_kwargs.get("local_model_dir", '/home/siebenschuh/AdaParse/models/facebook__nougat-base')  # path to pre-downloaded repo
    model_src        = local_model_dir
    dtype_str        = parser_kwargs.get("dtype", "bfloat16")
    device_override  = parser_kwargs.get("device", None)           # "cuda" / "cpu"
    batch_pages      = int(parser_kwargs.get("batch_pages", 6))
    render_dpi       = int(parser_kwargs.get("render_dpi", 192))
    gen_kwargs       = parser_kwargs.get("gen_kwargs", {}) or {}

    with Timer('initialize-hf-nougat', unique_id):
        processor, model, device, torch_dtype = _load_hf_nougat(model_src, dtype_str, device_override)

    # Process each PDF → list[str] (one per page), then wrap like your old schema
    with Timer('hf-parser-parse', unique_id):
        documents = []
        for pdf_path in pdf_paths:
            page_texts = _nougat_infer_pages(
                pdf_path,
                processor=processor,
                model=model,
                device=device,
                torch_dtype=torch_dtype,
                batch_pages=batch_pages,
                dpi=render_dpi,
                gen_kwargs=gen_kwargs,
            )
            documents.append({
                "source_path": pdf_path,
                "pages": [{"index": i, "text": t} for i, t in enumerate(page_texts)],
            })
    # X X X X X

    # If parsing failed, return early
    if documents is None:
        logger.info(f'Failed to parse {pdf_paths}')
        return

    # Write the parsed documents to a JSON lines file
    with Timer('write-jsonl', unique_id):
        # Merge parsed documents into a single string of JSON lines
        lines = ''.join(f'{json.dumps(doc)}\n' for doc in documents)

        # Store the JSON lines strings to a disk using a single write operation
        with open(output_dir / f'{parser.unique_id}.jsonl', 'a+') as f:
            f.write(lines)

    # Sometimes parsl won't flush the stdout, so this is necessary for logs
    print('', end='', flush=True)


def parse_zip(
    zip_file: str,
    tmp_storage: Path,
    output_dir: Path,
    parser_kwargs: dict[str, Any],
) -> None:
    """Parse the PDF files stored within a zip file.

    Parameters
    ----------
    zip_file : str
        Path to the zip file containing the PDFs to parse.
    tmp_storage : Path
        Path to the local storage directory.
    output_dir : Path
        Directory to write the output JSON lines file to.
    parser_kwargs : dict[str, Any]
        Keyword arguments to pass to the parser. Contains an extra `name`
        argument to specify the parser to use.
    """
    import shutil
    import subprocess
    import traceback
    import uuid
    from pathlib import Path

    from adaparse.convert import parse_pdfs
    from adaparse.timer import Timer

    # Time the worker function
    timer = Timer('finished-parsing', zip_file).start()

    try:
        # Make a temporary directory to unzip the file (use a UUID
        # to avoid name collisions)
        local_dir = tmp_storage / str(uuid.uuid4())
        temp_dir = local_dir / Path(zip_file).stem
        temp_dir.mkdir(parents=True)

        # Unzip the file (quietly--no verbose output)
        subprocess.run(['unzip', '-q', zip_file, '-d', temp_dir], check=False)

        # Glob the PDFs
        pdf_paths = [str(p) for p in temp_dir.glob('**/*.pdf')]

        # Call the parse_pdfs function
        with Timer('parse-pdfs', zip_file):
            parse_pdfs(pdf_paths, output_dir, parser_kwargs)

        # Clean up the temporary directory
        shutil.rmtree(local_dir)

    # Catch any exceptions possible. Note that we need to convert the exception
    # to a string to avoid issues with pickling the exception object
    except BaseException as e:
        if local_dir.exists():
            shutil.rmtree(local_dir)

        traceback.print_exc()
        print(f'Failed to process {zip_file}: {e}')
        return None

    finally:
        # Stop the timer to log the worker time and flush the buffer
        timer.stop(flush=True)


def parse_checkpoint(checkpoint_path: str) -> set[str]:
    """Parse which input paths have been completed from a adaparse output dir.

    NOTE: This function currently only is possible if the input is parsed with
    zip files. The raw pdf parsing logging does not log each individual pdf
    file parsed. If we need this functionality we need to explicitly log each
    parsed pdf instead of grepping the timing logs.

    Parameters
    ----------
    checkpoint_path : str
        Path to root adaparse directory. Should contain a `parsl` directory

    Returns
    -------
    set[str]
        A set of paths that have already been parsed in previous runs
    """
    # Grab time logger for parsing functionality
    from adaparse.timer import TimeLogger

    # get all *.stdout files
    stdout_files = Path(checkpoint_path).glob('**/*.stdout')

    # Find out which files have been successfully parsed by the workflow in
    # previous runs
    parsed_files = set()
    for stdout_file in stdout_files:
        time_stats = TimeLogger().parse_logs(stdout_file)
        for log_elem in time_stats:
            tags = log_elem.tags
            if 'finished-parsing' in tags:
                # This is will add everything after the tag type (first elem)
                # to the set. Currently there should only be one element after
                # but this will extend to more types of logs if they occur
                for elem in tags[1:]:
                    parsed_files.add(elem)

    return parsed_files


class WorkflowConfig(BaseModel):
    """Configuration for the PDF parsing workflow."""

    pdf_dir: Path
    """Directory containing pdfs to parse."""

    out_dir: Path
    """The output directory of the workflow."""

    tmp_storage: Path = Path('/tmp')
    """Temporary storage directory for unzipping files."""

    iszip: bool = False
    """Whether the input files are zip files containing many PDFs."""

    num_conversions: int = sys.maxsize
    """Number of pdfs to convert (useful for debugging)."""

    chunk_size: int = 1
    """Number of pdfs to convert in a single batch."""

    parser_settings: ParserConfigTypes
    """Parser settings (e.g., model paths, etc)."""

    compute_settings: ComputeSettingsTypes
    """Compute settings (HPC platform, number of GPUs, etc)."""


if __name__ == '__main__':
    parser = ArgumentParser(description='PDF conversion workflow')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to workflow configuration file',
    )
    args = parser.parse_args()

    # Load workflow configuration
    config = WorkflowConfig.from_yaml(args.config)

    # Setup output directory
    config.out_dir = config.out_dir.resolve()
    config.out_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    logger = setup_logging('adaparse', config.out_dir)

    logger.info(f'Loaded configuration: {config}')

    # Save the configuration to the output directory
    config.write_yaml(config.out_dir / 'config.yaml')

    # If we have run before, find all previously parsed files
    # else we use a empty set to check against
    # NOTE: this function assumes the input file paths have not changed from
    # run to run. If they have this method will fail and there will be
    # duplicated parses. Similarly, if you switch from parsing zips to parsing
    # pdfs it will fail.
    if (config.out_dir / 'parsl').exists():
        already_parsed_files = parse_checkpoint(str(config.out_dir / 'parsl'))
    else:
        already_parsed_files = set()

    # File extension for the input files
    file_ext = 'zip' if config.iszip else 'pdf'

    # Collect files and check if already parsed before
    files = [
        p.as_posix()
        for p in config.pdf_dir.glob(f'**/*.{file_ext}')
        if p.as_posix() not in already_parsed_files
    ]

    # Limit the number of conversions for debugging
    if len(files) >= config.num_conversions:
        files = files[: config.num_conversions]
        logger.info(
            f'len(files) exceeds {config.num_conversions}. '
            f'Only first {config.num_conversions} pdfs passed.'
        )

    # Log the input files
    logger.info(f'Found {len(files)} {file_ext} files to parse')

    # Batch the input args
    # Zip files have many PDFs, so we process them in a single batch,
    # while individual PDFs are batched in chunks to maintain higher throughput
    batched_files = (
        files if config.iszip else batch_data(files, config.chunk_size)
    )

    # Create a subdirectory to write the output to
    pdf_output_dir = config.out_dir / 'parsed_pdfs'
    pdf_output_dir.mkdir(exist_ok=True)

    # Log the output directory and number of batches
    logger.info(f'Writing output to {pdf_output_dir}')
    logger.info(f'Processing {len(batched_files)} batches')  # type: ignore[arg-type]

    # Ensure that AdaParse requires zipped PDFs
    if config.parser_settings.name == 'adaparse' and not config.iszip:
        raise ValueError(
            'AdaParse requires input PDFs to be provided as ZIP files.\n'
            'Consult https://github.com/7shoe/AdaParse?tab=readme-ov-file#data-preparation\n'
            'Set `iszip: true` in your YAML config.'
        )

    # Setup the worker function with default arguments
    if config.iszip:
        worker_fn = functools.partial(
            parse_zip,
            tmp_storage=config.tmp_storage,
            output_dir=pdf_output_dir,
            parser_kwargs=config.parser_settings.model_dump(),
        )
    else:
        worker_fn = functools.partial(
            parse_pdfs,
            output_dir=pdf_output_dir,
            parser_kwargs=config.parser_settings.model_dump(),
        )

    # Setup parsl for distributed computing
    parsl_config = config.compute_settings.get_config(config.out_dir / 'parsl')

    # Log the checkpoint files
    logger.info(
        f'Found the following checkpoints: {parsl_config.checkpoint_files}'
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, batched_files))
