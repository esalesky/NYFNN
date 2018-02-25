"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
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

    max_num_sents = 100  # high enough to get all sents
    max_sent_length = 50
    max_gen_length = 100

    # Load the model
    model = pickle.load(open(args.model, 'rb'))
    if use_cuda:
        model = model.cuda()
    src_vocab = pickle.load(open(args.srcvocab, 'rb'))
    tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))
    file_prefix = "data/{}/IWSLT16.TED.tst2012.{}".format(pair, pair)

    src_vocab, tgt_vocab, tst_sents = input_reader(file_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length,
                                                   src_vocab, tgt_vocab, file_suffix='.xml')

    loss_fn = nn.NLLLoss()
    generate(model, tst_sents, src_vocab, tgt_vocab, max_gen_length, loss_fn, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-s", "--srcvocab")
    parser.add_argument("-t", "--tgtvocab")
    parser.add_argument("-sl", "--srclang", default="en")
    parser.add_argument("-tl", "--tgtlang", default="cs")
    parser.add_argument("-o", "--output", default="gen-output.txt")
    args = parser.parse_args()
    main(args)
