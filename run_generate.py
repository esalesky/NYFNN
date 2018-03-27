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

def main(args):
    print("Use CUDA: {}".format(use_cuda))

    src_lang = args.srclang
    tgt_lang = args.tgtlang
    pair = "en-" + args.tgtlang

    max_sent_length = 50
    max_gen_length = 100
    max_num_sents = 100000

    debug = True
    if debug:
        train_prefix = 'data/examples/debug'
        dev_prefix = 'data/examples/debug'
        tst_prefix = 'data/examples/debug'
        file_suffix = ''
    else:
        train_prefix = 'data/{}/bped/train.tags.{}'.format(pair, pair)
        dev_prefix   = 'data/{}/bped/IWSLT16.TED.tst2012.{}'.format(pair, pair)
        tst_prefix   = 'data/{}/bped/IWSLT16.TED.tst2013.{}'.format(pair, pair)
        file_suffix  = ".bpe"

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

    trainer = MTTrainer(model, None, optim_type='Adam', batch_size=1, beam_size=5,
                        learning_rate=0.0001)
    avg_loss, total_loss = trainer.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, args.output, plot_attn=True)
    print("Total dev loss: {}, Average dev loss: {}".format(total_loss, avg_loss))


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
