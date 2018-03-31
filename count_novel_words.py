"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random
import pickle
import torch.nn as nn
import torch

# local imports
from preprocessing import input_reader
from utils import use_cuda, MODEL_PATH
from training import MTTrainer
import logging
import logging.config

logger = logging.getLogger(__name__)

def main(args):

    test_file = open(args.test)
    if args.ref == 'dev':
        ref_file = open("data/en-cs/IWSLT16.TED.tst2012.en-cs.cs.txt")
    elif args.ref == 'tst':
        ref_file = open("data/en-cs/IWSLT16.TED.tst2013.en-cs.cs.txt")
    src_lang = args.srclang
    tgt_lang = args.tgtlang
    pair = "en-" + args.tgtlang

    max_sent_length = 50
    max_num_sents = 250000

    src_suffix = ".txt"
    tgt_suffix = ".txt"

    train_src = 'data/{}/train.tags.{}.{}{}'.format(pair, pair, src_lang, src_suffix)
    train_tgt = 'data/{}/train.tags.{}.{}{}'.format(pair, pair, tgt_lang, tgt_suffix)

    src_vocab, tgt_vocab, train_sents = input_reader(train_src, train_tgt, src_lang, tgt_lang, max_num_sents,
                                                     max_sent_length, sort=True)

    test_lines = [line for line in test_file]
    ref_lines = [line for line in ref_file]

    spurious_tok = 0
    non_spurious_tok = 0
    spurious_forms = set()
    non_spurious_forms = set()
    for line_num, line in enumerate([pair for pair in zip(test_lines, ref_lines)]):
        test_line = line[0]
        ref_line = line[1]
        tokens = test_line.split(" ")
        for t in tokens:
            t = t.strip()
            token = tgt_vocab.map2idx(t)
            if token == tgt_vocab.unk_token:
                if t in ref_line:
                    non_spurious_tok += 1
                    non_spurious_forms.add(t)
                    print("Non-spurious unknown word generated {} in sentence {}".format(t, line_num + 1))
                else:
                    spurious_tok += 1
                    spurious_forms.add(t)
                    print("Spurious unknown word generated {} in sentence {}".format(t, line_num + 1))
    print("Non-spurious new tokens: {}, Non-Spurious new forms: {}\nSpurious new words: {}, Spurious new forms: {}"
            .format(non_spurious_tok, len(non_spurious_forms), spurious_tok, len(spurious_forms)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test")
    parser.add_argument("-r", "--ref", default="dev")
    parser.add_argument("-sl", "--srclang", default="en")
    parser.add_argument("-tl", "--tgtlang", default="cs")
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'gen.log'})
    main(args)
