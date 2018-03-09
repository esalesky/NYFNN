"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random
import pickle
import torch.nn as nn

# local imports
from preprocessing import input_reader
from encdec import RNNEncoder, RNNDecoder, EncDec, AttnDecoder
from utils import use_cuda, MODEL_PATH
from train_monitor import TrainMonitor

def main(args):
    print("Use CUDA: {}".format(use_cuda))  # currently always false, set in utils

    src_lang = args.srclang
    tgt_lang = args.tgtlang
    pair = "en-" + args.tgtlang

    max_sent_length = 50
    max_gen_length = 100

    # Load the model
    model = pickle.load(open(args.model, 'rb'))
    if use_cuda:
        model = model.cuda()
    src_vocab = pickle.load(open(args.srcvocab, 'rb'))
    tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))
    file_prefix = ".".join(args.input.split(".")[0:-2])

    src_vocab, tgt_vocab, tst_sents = input_reader(file_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length,
                                                   src_vocab, tgt_vocab, file_suffix='.txt')

    trainer = MTTrainer(model, monitor, optim_type='SGD', batch_size=1, learning_rate=0.01)
    trainer.generate(tst_sents, src_vocab, tgt_vocab, max_gen_length, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-i", "--input")
    parser.add_argument("-sv", "--srcvocab")
    parser.add_argument("-tv", "--tgtvocab")
    parser.add_argument("-sl", "--srclang", default="en")
    parser.add_argument("-tl", "--tgtlang", default="cs")
    parser.add_argument("-o", "--output", default="output/gen-output.txt")
    args = parser.parse_args()
    main(args)
