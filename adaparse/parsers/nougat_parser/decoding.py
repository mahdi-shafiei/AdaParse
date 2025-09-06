import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex
import numpy as np
from adaparse.parsers.nougat_parser.postprocessing import postprocess
from transformers import StoppingCriteria
from collections import defaultdict

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        # values has shape (batch, num_samples_in_window)
        n = 0 if self.values is None else self.values.shape[1]
        if n < 2:
            # Not enough samples yet → define variance as zeros (no early stopping impact)
            return torch.zeros(
                self.values.shape[0],
                dtype=self.values.dtype,
                device=self.values.device,
            )

        # Use sample variance (ddof=1) once we have >=2 samples (preserves your behavior)
        try:
            v = torch.var(self.values, dim=1, correction=1)  # PyTorch ≥1.10
        except TypeError:
            v = torch.var(self.values, dim=1, unbiased=True)  # older API

        return v / n if self.norm else v

# class RunningVarTorch
class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

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

        has_eos = indices[b].eq(tokenizer.eos_token_id).any().item()
        enough_room = (N + 1) < indices.shape[1]

        #if (indices[b] == tokenizer.eos_token_id).any() and (N + 1 < indices.shape[1]): # bug of .any()
        #if (indices[b] == tokenizer.eos_token_id) and (N + 1 < indices.shape[1]):
        if has_eos and enough_room:
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
