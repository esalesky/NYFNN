"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random
import pickle

#local imports
from preprocessing import input_reader
from encdec import RNNEncoder, RNNDecoder, EncDec
from training import train_setup, generate
from utils import use_cuda, MODEL_PATH


def main(args):
    print("Use CUDA: {}".format(use_cuda))  #currently always false, set in utils

    src_lang = 'cs'
    tgt_lang = 'en'
    # data_prefix = 'data/examples/debug'
    data_prefix = 'data/en-cs/train.tags.en-cs'
    
    max_sent_length = 50
    max_num_sents   = 100
    
    src_vocab, tgt_vocab, train_sents = input_reader(data_prefix, src_lang, tgt_lang, max_sent_length, max_num_sents)

    hidden_size = 64
    input_size  = src_vocab.vocab_size()
    output_size = tgt_vocab.vocab_size()

    # Initialize our model
    if args.model is not None:
        model = pickle.load(open(args.model, 'rb'))
    else:
        enc = RNNEncoder(vocab_size=input_size, embed_size=hidden_size, hidden_size=hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False)
        dec = RNNDecoder(vocab_size=output_size, embed_size=hidden_size, hidden_size=hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False)

        if use_cuda:
            enc = enc.cuda()
            dec = dec.cuda()

        model = EncDec(enc, dec)

    train_setup(model, train_sents, num_epochs=30, print_every=5, plot_every=5)
    generate(model, train_sents, src_vocab, tgt_vocab, max_sent_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()
    main(args)

