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

    src_lang = 'en'
    tgt_lang = 'cs'
    pair = 'en-cs'
#    pair = 'en-de'

    train_prefix = 'data/'+pair+'/train.tags.'+pair
    dev_prefix   = 'data/'+pair+'/IWSLT16.TED.tst2012.'+pair
    tst_prefix   = 'data/'+pair+'/IWSLT16.TED.tst2013.'+pair
#    train_prefix = 'data/examples/debug'
    
    max_num_sents   = 60000  #high enough to get all sents
    max_sent_length = 30
    num_epochs  = 30
    print_every = 50
    plot_every  = 50
    model_every = 10000
    
    src_vocab, tgt_vocab, train_sents = input_reader(train_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length)
    src_vocab, tgt_vocab, dev_sents   = input_reader(dev_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, src_vocab, tgt_vocab, file_suffix='.xml')
    src_vocab, tgt_vocab, tst_sents   = input_reader(tst_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, src_vocab, tgt_vocab, file_suffix='.xml')

    hidden_size = 128
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

    train_setup(model, train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab,
                num_epochs=num_epochs, print_every=print_every, plot_every=plot_every, model_every=model_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()
    main(args)
