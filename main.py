"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import pickle
import torch
import random

from encdec import RNNEncoder, RNNDecoder, EncDec, AttnDecoder
from preprocessing import input_reader
from utils import use_cuda, MODEL_PATH
from training import MTTrainer
import logging
import logging.config
from train_monitor import TrainMonitor
import torch
import random

def main(args):
    logger = logging.getLogger(__name__)
    logger.info("Use CUDA: {}".format(use_cuda))  #currently always false, set in utils

    src_lang = 'en'
    tgt_lang = 'cs'  #cs or de
    pair = "en-" + tgt_lang

    debug=False
    fixed_seeds=True
    if debug:
        train_prefix = 'data/examples/debug'
        dev_prefix = 'data/examples/debug'
        tst_prefix = 'data/examples/debug'
        file_suffix = ''
    else:
        train_prefix = 'data/{}/bped/train.tags.{}'.format(pair, pair)
        dev_prefix   = 'data/{}/bped/IWSLT16.TED.tst2012.{}'.format(pair, pair)
        tst_prefix   = 'data/{}/bped/IWSLT16.TED.tst2013.{}'.format(pair, pair)
        file_suffix  = ".tok.bpe"
    if fixed_seeds:
        torch.manual_seed(69)
        if use_cuda:
            torch.cuda.manual_seed(69)
        random.seed(69)
    
    max_num_sents = int(args.maxnumsents)
    batch_size = 80
    max_sent_length = 50  #paper: 50 for baseline, 100 for morphgen
    max_gen_length  = 100    
    num_epochs  = 30
    print_every = 50
    plot_every  = 50
    model_every = 500
    bi_enc = True
    # Encoder and decoder hidden size must change together
    enc_hidden_size = 1024
    if bi_enc:
        enc_hidden_size = int(enc_hidden_size / 2)
    dec_hidden_size = 1024  #paper: 1024
    embed_size  = 500   #paper: 500
    
    src_vocab, tgt_vocab, train_sents = input_reader(train_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, file_suffix=file_suffix, sort=True)
    src_vocab, tgt_vocab, dev_sents   = input_reader(dev_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, src_vocab, tgt_vocab, file_suffix=file_suffix, filt=False)
    src_vocab, tgt_vocab, tst_sents   = input_reader(tst_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, src_vocab, tgt_vocab, file_suffix=file_suffix, filt=False)

    input_size  = src_vocab.vocab_size()
    output_size = tgt_vocab.vocab_size()    

    # Initialize our model
    if args.model is not None:
        model = pickle.load(open(args.model, 'rb'))
        src_vocab = pickle.load(open(args.srcvocab, 'rb'))
        tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))
    else:
        src_vocab.save("models/src-vocab_" + pair + "_maxnum" + str(max_num_sents) +
                       "_maxlen" + str(max_sent_length) + ".pkl")
        tgt_vocab.save("models/tgt-vocab_" + pair + "_maxnum" + str(max_num_sents) +
                       "_maxlen" + str(max_sent_length) + ".pkl")

        enc = RNNEncoder(vocab_size=input_size, embed_size=embed_size,
                         hidden_size=enc_hidden_size, rnn_type='GRU',
                         num_layers=1, bidirectional=bi_enc)
        dec = AttnDecoder(enc_size=enc_hidden_size,vocab_size=output_size,
                          embed_size=embed_size, hidden_size=dec_hidden_size,
                          rnn_type='GRU', num_layers=1, bidirectional_enc=bi_enc,
                          tgt_vocab=tgt_vocab)
        # dec = RNNDecoder(vocab_size=output_size, embed_size=embed_size, hidden_size=hidden_size, rnn_type='GRU', num_layers=1, bidirectional=False)
        model = EncDec(enc, dec)

    if use_cuda:
        model = model.cuda()

    monitor = TrainMonitor(model, len(train_sents), print_every=print_every,
                           plot_every=plot_every, save_plot_every=plot_every,
                           checkpoint_every=model_every)

    trainer = MTTrainer(model, monitor, optim_type='Adam', batch_size=batch_size,
                        learning_rate=0.0001)

    trainer.train(train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab, num_epochs,
                  max_gen_length=max_gen_length, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-s", "--srcvocab", default=None)
    parser.add_argument("-t", "--tgtvocab", default=None)
    parser.add_argument("-n", "--maxnumsents", default=250000)  #defaults to high enough for all
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'training.log'})
    main(args)
