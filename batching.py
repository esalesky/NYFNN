"""Code for batching examples by length."""
from collections import Counter
import numpy as np
import itertools
import random

# Local imports
from preprocessing import input_reader, EOS


def make_batches(sent_pairs, batch_size=64):
    """Make batches for the given sentences.

    All batches have source sentences of the same length. (still true)
    Target sentences are padded with EOS symbol.
    """
    random.shuffle(sent_pairs) #randomly shuffle sent pairs
    sent_pairs = sorted(sent_pairs, key=lambda x: x[0]) #and then only sort by src, not tgt
    
    batch_idxs = get_batch_idxs(sent_pairs, batch_size)

    batches = []
    for b in batch_idxs:
        sents = sent_pairs[b[0]:b[1]]
        src = np.array([pair[0] for pair in sents])
        tgt = pad_sents([pair[1] for pair in sents])
        batches.append((src, tgt))

    return batches


def get_batch_idxs(sent_pairs, batch_size):
    """Get the indices for splitting out batches."""
    last_i = 0
    batch_idxs = []
    total_sentences = 0

    # Count the lengths of all source sentences.
    src_len_counts = Counter(map(len, sent_pairs))

    # Go through and find the indexes to make batches <= batch_size
    for n_words, n_sentences in src_len_counts.items():
        total_sentences += n_sentences
        while(last_i + batch_size < total_sentences):
            batch_idxs.append((last_i, last_i + batch_size))
            last_i += batch_size
        batch_idxs.append((last_i, total_sentences))
        last_i = total_sentences

    return batch_idxs

def pad_sents(sents, fillval=EOS):
    """Pad sentences with trailing values."""
    return np.array(list(itertools.zip_longest(*sents, fillvalue=fillval))).T
