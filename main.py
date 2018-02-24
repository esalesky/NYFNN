"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random

#local imports
from preprocessing import input_reader
from encdec import RNNEncoder, RNNDecoder, EncDec
from training import train_setup
from utils import use_cuda


def main():
    print("Use CUDA: {}".format(use_cuda))  #currently always false, set in utils

    src_lang = 'en'
    tgt_lang = 'de'
    data_prefix = 'data/examples/debug'

    max_sent_length = 10
    max_num_sents   = 10000
    
    src_vocab, tgt_vocab, train_sents = input_reader(data_prefix, src_lang, tgt_lang, max_sent_length, max_num_sents)

    hidden_size = 64
    input_size  = src_vocab.vocab_size()
    output_size = tgt_vocab.vocab_size()

    #-------------------------------------

    enc = RNNEncoder(vocab_size=input_size, embed_size=hidden_size, hidden_size=hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False)
    dec = RNNDecoder(vocab_size=output_size, embed_size=hidden_size, hidden_size=hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False)

    if use_cuda:
        enc = enc.cuda()
        dec = dec.cuda()

    model = EncDec(enc, dec)

    train_setup(model, train_sents, num_epochs=30, print_every=1)

#    model.save('model.pkl') #todo: move to train_setup? save every epoch?

    pass


if __name__ == "__main__":
    main()

