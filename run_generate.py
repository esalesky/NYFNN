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
    logger.info("Use CUDA: {}".format(use_cuda))

    src_lang = args.srclang
    tgt_lang = args.tgtlang
    pair = "en-" + args.tgtlang

    max_sent_length = 50
    max_gen_length = 100
    max_num_sents = 100000

    debug = True
    if debug:
        train_src = 'data/examples/debug.en'
        train_tgt = 'data/examples/debug.cs'
        dev_src = 'data/examples/debug.en'
        dev_tgt = 'data/examples/debug.cs'
        tst_src = 'data/examples/debug.en'
        tst_tgt = 'data/examples/debug.cs'
    else:
        src_dir = "bped"
        tgt_dir = "bped"
        #        tgt_dir = "morphgen_bpe"
        src_suffix = ".tok.bpe"
        tgt_suffix = ".tok.bpe"
        #        tgt_suffix = "-morph.bpe"

        train_src = 'data/{}/{}/train.tags.{}.{}{}'.format(pair, src_dir, pair, src_lang, src_suffix)
        train_tgt = 'data/{}/{}/train.tags.{}.{}{}'.format(pair, tgt_dir, pair, tgt_lang, tgt_suffix)
        dev_src = 'data/{}/{}/IWSLT16.TED.tst2012.{}.{}{}'.format(pair, src_dir, pair, src_lang, src_suffix)
        dev_tgt = 'data/{}/{}/IWSLT16.TED.tst2012.{}.{}{}'.format(pair, tgt_dir, pair, tgt_lang, tgt_suffix)
        tst_src = 'data/{}/{}/IWSLT16.TED.tst2013.{}.{}{}'.format(pair, src_dir, pair, src_lang, src_suffix)
        tst_tgt = 'data/{}/{}/IWSLT16.TED.tst2013.{}.{}{}'.format(pair, tgt_dir, pair, tgt_lang, tgt_suffix)

    # Load the model
    model = torch.load(args.model)
    if use_cuda:
        model = model.cuda()
    logger.info("Loaded model")
    model.eval()
    src_vocab = pickle.load(open(args.srcvocab, 'rb'))
    tgt_vocab = pickle.load(open(args.tgtvocab, 'rb'))

    if args.input == 'dev':
        src_vocab, tgt_vocab, sents = input_reader(dev_src, dev_tgt, src_lang, tgt_lang, max_num_sents,
                                                       max_sent_length, src_vocab, tgt_vocab, filt=False)

    elif args.input =='tst':
        src_vocab, tgt_vocab, sents = input_reader(tst_src, tst_tgt, src_lang, tgt_lang, max_num_sents,
                                                       max_sent_length, src_vocab, tgt_vocab, filt=False)

    else:
        logger.error("Unrecognized input type: {}".format(args.input))
        return

    trainer = MTTrainer(model, None, optim_type='Adam', batch_size=1, beam_size=5,
                        learning_rate=0.0001)
    avg_loss, total_loss = trainer.generate(sents, src_vocab, tgt_vocab, max_gen_length, args.output, plot_attn=True)
    logger.info("Total dev loss: {}, Average dev loss: {}".format(total_loss, avg_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-i", "--input", default="dev")
    parser.add_argument("-sv", "--srcvocab")
    parser.add_argument("-tv", "--tgtvocab")
    parser.add_argument("-sl", "--srclang", default="en")
    parser.add_argument("-tl", "--tgtlang", default="cs")
    parser.add_argument("-o", "--output", default="gen-output.txt")
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'gen.log'})
    main(args)
