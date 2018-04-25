"""read data, create vocab, and preprocess"""
from sortedcontainers import SortedList
from collections import defaultdict
import unicodedata
import pickle
try:
    import regex as re
except ImportError:
    import re

import logging

SOS = 0
EOS = 1

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


#todo: tokenization, bpe

logger = logging.getLogger(__name__)

class Vocab:
    """Class for mapping source and target vocab to indices."""

    def __init__(self, name):
        self.name = name
        self.word2idx = {SOS_TOKEN : SOS, EOS_TOKEN : EOS}
        self.idx2word = [SOS_TOKEN, EOS_TOKEN]

        self.unk_token  = None     #default None, set below after vocab frozen
        self.vocab_frozen = False  #impt for decoding, where we should not add new vocab

    def vocab_size(self):
        return len(self.idx2word)

    # maps words to vocab idx, adds if not already present
    def map2idx(self, word):
        if word not in self.word2idx:
            if self.vocab_frozen:
                assert self.unk_token != None, 'No unk_token set but tried to map OOV word to idx with frozen vocab'
                return self.unk_token
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def freeze_vocab(self):
        self.vocab_frozen = True

    def set_unk(self, unk_word):
        assert self.vocab_frozen, 'Tried to set unk with vocab not frozen'
        if unk_word not in self.word2idx:
            self.word2idx[unk_word] = len(self.idx2word)
            self.idx2word.append(unk_word)
        self.unk_token = self.word2idx[unk_word]

    def save(self, fname):
        """Save the vocabulary to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

class SentencePair:
    """Class for source, target sentence pairs. Allows easy sorting for minibatching."""

    def __init__(self, pair):
        self.pair = pair

    def __len__(self):
        return len(self[0])

    def __repr__(self):
        return "Sentence Pair: " + self.pair.__repr__()

    def __getitem__(self, i):
        return self.pair[i]

    def __lt__(self, other):
        if len(self) != len(other):
            return len(self) < len(other)
        else:
            # Defer to tgt sent length if src lengths are equal
            return len(self[1]) < len(other[1])

    def __gt__(self, other):
        if len(self) != len(other):
            return len(self) > len(other)
        else:
            # Defer to tgt sent length if src lengths are equal
            return len(self[1]) > len(other[1])



# reads parallel data where format is one sentence per line, filename prefix.lang
# expectation is file does not have SOS/EOS symbols
def read_corpus(source_file, target_file, src_lang, tgt_lang, max_num_sents, src_vocab, tgt_vocab, max_sent_length, min_sent_length, sort, filt):
    src_file = source_file
    tgt_file = target_file

    logger.info("Reading files... src: {}, tgt: {}".format(src_file, tgt_file))

    sents = []
    with open(src_file, encoding='utf-8') as src_sents, open(tgt_file, encoding='utf-8') as tgt_sents:
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            src_sent = clean(src_sent)
            tgt_sent = clean(tgt_sent)
            if not filt or keep_pair((src_sent, tgt_sent), max_sent_length, min_sent_length):
                src_sent = [SOS] + [src_vocab.map2idx(w) for w in src_sent] + [EOS]
                tgt_sent = [tgt_vocab.map2idx(w) for w in tgt_sent] + [EOS]
                sents.append((src_sent, tgt_sent))
                if len(sents) >= max_num_sents:
                    break

    logger.info("Read {} sentences.".format(len(sents)))

    # Sort the sentence pairs
    if sort:
        sents = SortedList([*map(SentencePair, sents)])[:]
    else:
        sents = [*map(SentencePair, sents)]

    return sents


def create_vocab(source_file, target_file, src_lang, tgt_lang, max_num_sents,
                 max_sent_length=100, min_sent_length=2, max_vocab_size=50000):

    src_freq = defaultdict(int)
    tgt_freq = defaultdict(int)
    
    src_vocab = Vocab(src_lang)
    tgt_vocab = Vocab(tgt_lang)

    logger.info("Creating vocabs.")

    sent_counter = 0
    with open(source_file, encoding='utf-8') as src_sents, open(target_file, encoding='utf-8') as tgt_sents:
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            src_sent = clean(src_sent)
            tgt_sent = clean(tgt_sent)
            if keep_pair((src_sent, tgt_sent), max_sent_length, min_sent_length):
                src_freq[SOS_TOKEN]+=1
                src_freq[EOS_TOKEN]+=1
                tgt_freq[SOS_TOKEN]+=1
                tgt_freq[EOS_TOKEN]+=1
                for w in src_sent:
                    src_freq[w]+=1
                for w in tgt_sent:
                    tgt_freq[w]+=1

                sent_counter+=1
                if sent_counter >= max_num_sents:
                    break

    #---
    for v in sorted(src_freq, key=lambda x: src_freq[x], reverse=True)[:max_vocab_size]:
        src_vocab.map2idx(v)
    for v in sorted(tgt_freq, key=lambda x: tgt_freq[x], reverse=True)[:max_vocab_size]:
        tgt_vocab.map2idx(v)

    src_vocab.freeze_vocab()
    src_vocab.set_unk(UNK_TOKEN)

    tgt_vocab.freeze_vocab()
    tgt_vocab.set_unk(UNK_TOKEN)

    logger.info("Vocabs created.")

    return src_vocab, tgt_vocab


def clean(line):
    if line.startswith('<'):  # Handles the IWSLT XML formats
        if line.startswith('<seg id'):
            line = re.sub('<[^>]*>', '', line).strip()
        else:
            line = ""
    else:
        line = line.strip()  # Supports both txt and the pseudo IWSLT XML
    return line.split()


def input_reader(source_file, target_file, src, tgt, max_num_sents,
                 max_sent_length=100, src_vocab=None, tgt_vocab=None, sort=False, filt=True):

    sents = read_corpus(source_file, target_file, src, tgt, max_num_sents,
                        src_vocab, tgt_vocab, max_sent_length, min_sent_length=1, sort=sort, filt=filt)

    return sents
    

#true if sent should not be filtered
def keep_pair(p, max_sent_length, min_sent_length):
    return min_sent_length <= len(p[0]) <= max_sent_length and min_sent_length <= len(p[1]) <= max_sent_length
