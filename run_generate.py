"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random
import pickle
import torch.nn as nn
import torch

# local imports
from preprocessing import input_reader
from encdec import RNNEncoder, RNNDecoder, EncDec, AttnDecoder
from utils import use_cuda, MODEL_PATH
from train_monitor import TrainMonitor
from training import MTTrainer

def main(args):
    print("Use CUDA: {}".format(use_cuda))

    src_lang = args.srclang
    tgt_lang = args.tgtlang
    pair = "en-" + args.tgtlang

    max_sent_length = 50
    max_gen_length = 100
    max_num_sents = 100000

    train_prefix = 'data/examples/debug'
    dev_prefix = 'data/examples/debug'
    tst_prefix = 'data/examples/debug'
    file_suffix = ''

    # Load the model
    model = torch.load(args.model)
    if use_cuda:
        model = model.cuda()
    print("Loaded model")
    model.eval()
    src_vocab = pickle.load(open(args.srcvocab, 'rb'))
    tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))

    src_vocab, tgt_vocab, dev_sents   = input_reader(dev_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length,
                                                     src_vocab, tgt_vocab, file_suffix=file_suffix, filt=False)


    trainer = MTTrainer(model, None, optim_type='Adam', batch_size=1,
                        learning_rate=0.0001)
    trainer.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-i", "--input")
    parser.add_argument("-sv", "--srcvocab")
    parser.add_argument("-tv", "--tgtvocab")
    parser.add_argument("-sl", "--srclang", default="en")
    parser.add_argument("-tl", "--tgtlang", default="cs")
    parser.add_argument("-o", "--output", default="gen-output.txt")
    args = parser.parse_args()
    main(args)
