"""Application for checking the size of a vocabulary and identifying the number of words from a source set of files
that are mapped to unk, based on that vocabulary."""
import argparse
import random
import pickle
import torch.nn as nn

# local imports
from preprocessing import input_reader
from encdec import RNNEncoder, RNNDecoder, EncDec
from training import train_setup, generate
from utils import use_cuda, MODEL_PATH

def main(args):
    print("Use CUDA: {}".format(use_cuda))  # currently always false, set in utils

    src_lang = args.srclang
    tgt_lang = args.tgtlang
    pair = args.srclang + "-" + args.tgtlang

    max_num_sents = 60000  # high enough to get all sents
    max_sent_length = 30
    max_gen_length = 100

    # Load the model
    src_vocab = pickle.load(open(args.srcvocab, 'rb'))
    tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))
    file_prefix = "data/{}/IWSLT16.TED.tst2012.{}".format(pair, pair)

    # src_size = src_vocab.vocab_size()
    # tgt_size = tgt_vocab.vocab_size()
    src_vocab, tgt_vocab, tst_sents = input_reader(file_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length,
                                                   src_vocab, tgt_vocab, file_suffix='.txt')
    # assert src_size == src_vocab.vocab_size()
    # assert tgt_size == tgt_vocab.vocab_size()

    src_unk = src_vocab.word2idx["<unk>"]
    tgt_unk = tgt_vocab.word2idx["<unk>"]

    print("Size of source vocabulary: {} ".format(src_vocab.vocab_size()))
    print("Size of target vocabulary: {}".format(tgt_vocab.vocab_size()))

    tok_count = [0, 0]
    unk_tok_count = [0, 0]

    src_types = set()
    tgt_types = set()

    for sent in tst_sents:
        src_sent, tgt_sent = sent
        for word_idx in src_sent:
            src_types.add(word_idx)
            if word_idx == src_unk:
                unk_tok_count[0] += 1
            tok_count[0] += 1
        for word_idx in tgt_sent:
            tgt_types.add(word_idx)
            if word_idx == tgt_unk:
                unk_tok_count[1] += 1
            tok_count[1] += 1

    # for i in src_vocab.word2idx.values():
    #     if i not in src_types:
    #         print(src_vocab.idx2word[i], i)

    print("Number of source types {}, Number of target types {}".format(len(src_types), len(tgt_types)))

    print("Total source tokens: {}, source unk tokens: {}, percent unk: {}".format(tok_count[0], unk_tok_count[0],
           float(unk_tok_count[0]) / tok_count[0]))
    print("Total target tokens: {}, target unk tokens: {}, percent unk: {}".format(tok_count[1], unk_tok_count[1],
          float(unk_tok_count[1]) / tok_count[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--srcvocab")
    parser.add_argument("-t", "--tgtvocab")
    parser.add_argument("-sl", "--srclang", default="en")
    parser.add_argument("-tl", "--tgtlang", default="cs")
    args = parser.parse_args()
    main(args)
