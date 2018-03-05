"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import pickle

from encdec import RNNEncoder, RNNDecoder, EncDec
# local imports
from preprocessing import input_reader
from utils import use_cuda
from training import MTTrainer
import logging
import logging.config
from train_monitor import TrainMonitor

def main(args):
    logger = logging.getLogger(__name__)

    logger.info("Use CUDA: {}".format(use_cuda))  #currently always false, set in utils

    src_lang = 'en'
    tgt_lang = 'cs'  #cs or de
    pair = "en-" + tgt_lang

    train_prefix = 'data/'+pair+'/train.tags.'+pair
    dev_prefix   = 'data/'+pair+'/IWSLT16.TED.tst2012.'+pair
    tst_prefix   = 'data/'+pair+'/IWSLT16.TED.tst2013.'+pair
#    train_prefix = 'data/examples/debug'
    file_suffix  = ".txt"
    
    max_num_sents   = 100
    max_sent_length = 50
    max_gen_length = 100
    num_epochs  = 30
    print_every = 50
    plot_every  = 50
    model_every = 20000
    hidden_size = 128
    
    src_vocab, tgt_vocab, train_sents = input_reader(train_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, file_suffix=file_suffix)
    src_vocab, tgt_vocab, dev_sents   = input_reader(dev_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, src_vocab, tgt_vocab, file_suffix=file_suffix)
    src_vocab, tgt_vocab, tst_sents   = input_reader(tst_prefix, src_lang, tgt_lang, max_num_sents, max_sent_length, src_vocab, tgt_vocab, file_suffix=file_suffix)

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

        enc = RNNEncoder(vocab_size=input_size, embed_size=hidden_size, hidden_size=hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False)
        dec = RNNDecoder(vocab_size=output_size, embed_size=hidden_size, hidden_size=hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False)

        model = EncDec(enc, dec)

    if use_cuda:
        model = model.cuda()

    monitor = TrainMonitor(model, len(train_sents), print_every=print_every, plot_every=plot_every, save_plot_every=plot_every,
                           checkpoint_every=model_every)

    trainer = MTTrainer(model, monitor, optim_type='SGD', learning_rate=0.01)

    trainer.train(train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab, num_epochs, max_gen_length=max_gen_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-s", "--srcvocab", default=None)
    parser.add_argument("-t", "--tgtvocab", default=None)
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'training.log'})
    main(args)
