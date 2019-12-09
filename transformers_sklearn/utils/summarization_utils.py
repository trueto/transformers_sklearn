import os
import copy
import json
import torch
import logging
from torch.utils.data import Dataset
from .data_utils import to_numpy

logger = logging.getLogger(__name__)

class SummarizationDataset(Dataset):

    def __init__(self,X,y):
        self.stories = to_numpy(X)
        self.summaries = to_numpy(y)

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, item):
        story_lines = self.stories[item]
        summary_lines = self.summaries[item]
        return story_lines,summary_lines

# --------------------------
# Encoding and preprocessing
# --------------------------

def encode_for_summarization(story_lines, summary_lines, tokenizer):
    """ Encode the story and summary lines, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [tokenizer.encode(line) for line in story_lines]
    story_token_ids = [
        token for sentence in story_lines_token_ids for token in sentence
    ]
    summary_lines_token_ids = [tokenizer.encode(line) for line in summary_lines]
    summary_token_ids = [
        token for sentence in summary_lines_token_ids for token in sentence
    ]

    return story_token_ids, summary_token_ids

def fit_to_block_size(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding token to the right of the sequence.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence

def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask

def build_lm_labels(sequence, pad_token_id):
    """ Padding token are replaced by the value -1 so they
    are not taken into account in the loss computation. """
    padded = sequence.clone()
    padded[padded == pad_token_id] = -1
    return padded


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]

    The values {0,1} were found in the repository [2].

    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    for sequence in batch:
        sentence_num = -1
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)