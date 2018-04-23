"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import pickle
import torch
import random
import logging
import logging.config

#local imports
from encdec import RNNEncoder, RNNDecoder, EncDec, AttnDecoder, CondGruDecoder
from preprocessing import input_reader
from utils import use_cuda
from training import MTTrainer
from train_monitor import TrainMonitor


def main(args):
    params = __import__(args.config.replace('.py',''))

    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': '{}/training.log'.format(params.OUTPUT_PATH)})
    logger = logging.getLogger(__name__)
    logger.info("Use CUDA: {}".format(use_cuda))  #set automatically in utils

    if args.debug:
        params.train_src = 'data/examples/debug.en'
        params.train_tgt = 'data/examples/debug.cs'
        params.dev_src   = 'data/examples/debug.en'
        params.dev_tgt   = 'data/examples/debug.cs'
        params.tst_src   = 'data/examples/debug.en'
        params.tst_tgt   = 'data/examples/debug.cs'

    if params.fixed_seeds:
        torch.manual_seed(69)
        if use_cuda:
            torch.cuda.manual_seed(69)
        random.seed(69)
    
    max_num_sents = int(args.maxnumsents)
    
    src_vocab, tgt_vocab, train_sents = input_reader(params.train_src, params.train_tgt, params.src_lang, params.tgt_lang, max_num_sents, params.max_sent_length, sort=True)
    src_vocab, tgt_vocab, dev_sents_unsorted = input_reader(params.dev_src, params.dev_tgt, params.src_lang, params.tgt_lang, max_num_sents, params.max_sent_length,
                                                            src_vocab, tgt_vocab, filt=False)
    src_vocab, tgt_vocab, dev_sents_sorted   = input_reader(params.dev_src, params.dev_tgt, params.src_lang, params.tgt_lang, max_num_sents, params.max_sent_length,
                                                            src_vocab, tgt_vocab, sort=True, filt=False)
    src_vocab, tgt_vocab, tst_sents          = input_reader(params.tst_src, params.tst_tgt, params.src_lang, params.tgt_lang, max_num_sents, params.max_sent_length,
                                                            src_vocab, tgt_vocab, filt=False)

    input_size  = src_vocab.vocab_size()
    output_size = tgt_vocab.vocab_size()
    logger.info("src vocab size: {}".format(input_size)
    logger.info("tgt vocab size: {}".format(output_size)

    # Initialize our model
    if args.model is not None:
        model = torch.load(args.model)
        src_vocab = pickle.load(open(args.srcvocab, 'rb'))
        tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))
    else:
        src_vocab.save(params.MODEL_PATH + "src-vocab_" + params.pair + "_maxnum" + str(max_num_sents) +
                       "_maxlen" + str(params.max_sent_length) + ".pkl")
        tgt_vocab.save(params.MODEL_PATH + "tgt-vocab_" + params.pair + "_maxnum" + str(max_num_sents) +
                       "_maxlen" + str(params.max_sent_length) + ".pkl")

        enc = RNNEncoder(vocab_size=input_size, embed_size=params.embed_size,
                         hidden_size=params.enc_hidden_size, rnn_type='GRU',
                         num_layers=1, bidirectional=params.bi_enc)
        if params.cond_gru_dec:
            dec = CondGruDecoder(enc_size=params.enc_hidden_size, vocab_size=output_size,
                                 embed_size=params.embed_size, hidden_size=params.dec_hidden_size, bidirectional_enc=params.bi_enc)
        else:
            dec = AttnDecoder(enc_size=params.enc_hidden_size, vocab_size=output_size,
                              embed_size=params.embed_size, hidden_size=params.dec_hidden_size,
                              rnn_type='GRU', num_layers=1, bidirectional_enc=params.bi_enc,
                              tgt_vocab=tgt_vocab)

        model = EncDec(enc, dec)

    if use_cuda:
        model = model.cuda()

    monitor = TrainMonitor(model, len(train_sents), print_every=params.print_every,
                           plot_every=params.plot_every, save_plot_every=params.plot_every, model_every=params.model_every,
                           checkpoint_every=params.checkpoint_every, patience=params.patience, num_epochs=params.num_epochs,
                           output_path=params.OUTPUT_PATH, model_path=params.MODEL_PATH)

    trainer = MTTrainer(model, monitor, optim_type='Adam', batch_size=params.batch_size,
                        beam_size=params.beam_size, learning_rate=0.0001)

    trainer.train(train_sents, dev_sents_sorted, dev_sents_unsorted, tst_sents, src_vocab, tgt_vocab, params.num_epochs,
                  max_gen_length=params.max_gen_length, debug=args.debug, output_path=params.OUTPUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-s", "--srcvocab", default=None)
    parser.add_argument("-t", "--tgtvocab", default=None)
    parser.add_argument("-n", "--maxnumsents", default=250000)  #defaults to high enough for all
    parser.add_argument("-c", "--config", default="params.py")
    args = parser.parse_args()
    main(args)
