import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex
import numpy as np
from adaparse.parsers.nougat_parser.postprocessing import postprocess

def batch(l, b=15):
    """Helper: batching
    """
    subs = []
    for i in range(len(l) - b):
        subs.append(l[i : i + b])
    return subs

def subdiv(l, b=10):
    """Helper: sub-dividing
    """
    subs = []
    for i in range(len(l) - b):
        subs.append(l[: i + b])
    return subs

def process_decoder_output(decoder_output,
                           tokenizer,
                           early_stopping:bool=True,
                           min_len:int=120,
                           repeat_threshold:int=10,
                           variance_threshold:int=0.045,
                           small_var_threshold:int=30,
                           variance_mult:int=1.08,
                           max_id:int=4095):
    """
    Helper to post-process generated output
    """

    # output
    output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
        }

    output["repetitions"] = decoder_output.sequences.clone()
    output["sequences"] = decoder_output.sequences.clone()
    batch_size = len(decoder_output.sequences)

    # infer logits
    logits = torch.stack(decoder_output.scores, 1).cpu().max(-1)
    values = logits.values
    indices = logits.indices

    # loop
    for b in range(batch_size):
        mask = indices[b] != tokenizer.pad_token_id
        N = mask.sum().item()
        var = np.array(
            [np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())]
        )
        if len(var) < repeat_threshold:
            output["repeats"].append(None)
            continue
        varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1])

        #minlen = 120
        #if (indices[b] == tokenizer.eos_token_id).any() and (N + 1 < indices.shape[1]): # bug of .any()
        if (indices[b] == tokenizer.eos_token_id) and (N + 1 < indices.shape[1]):
            # there is an end to the generation, likely no repetitions
            output["repeats"].append(None)
            continue
        small_var = np.where(varvar < variance_threshold)[0]

        # early stopping
        if early_stopping and len(small_var) > 1:
            if np.all(np.diff(small_var) < 2):
                idx = int(min(max(small_var[0], 1) * variance_mult + min_len, max_id))
                if idx / N > 0.9:  # at most last bit
                    output["repeats"].append(None)
                    continue
                elif small_var[0] < small_var_threshold:
                    idx = 0
                #logging.warn("Found repetitions in sample %i" % b)
                output["repeats"].append(idx)
                output["sequences"][b, idx:] = tokenizer.pad_token_id
                output["repetitions"][b, :idx] = tokenizer.pad_token_id
            else:
                output["repeats"].append(None)
        else:
            output["repeats"].append(None)
    # loop done - - - -

    # batch decode
    output["repetitions"] = tokenizer.batch_decode(
        output["repetitions"], skip_special_tokens=True
    )
    # postprocess predictions
    output["predictions"] = postprocess(
        tokenizer.batch_decode(
            output["sequences"], skip_special_tokens=True
        ),
        markdown_fix=False,
    )

    return output
