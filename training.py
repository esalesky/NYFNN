"""training fns"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#local imports
from utils import use_cuda, pair2var, OUTPUT_PATH
from preprocessing import SOS, EOS
from batching import make_batches

import logging

logger = logging.getLogger(__name__)

def optimizer_factory(optim_type, model, **kwargs):
    assert optim_type in ['SGD', 'Adam'], 'Optimizer type not one of currently supported options'
    return getattr(optim, optim_type)(model.parameters(), **kwargs)


class MTTrainer:

    def __init__(self, model, train_monitor, optim_type='SGD', learning_rate=0.01):
        self.model = model
        self.optimizer = optimizer_factory(optim_type, model, lr=learning_rate)
        self.loss_fn = nn.NLLLoss()
        self.use_nllloss = True
        self.monitor = train_monitor

    def train_step(self, src, tgt, max_length):

        self.optimizer.zero_grad()
        loss = 0.0
        tgt_length = tgt.size()[0]

        decoder_scores, words = self.model(src, tgt)

        for gen, ref in zip(decoder_scores, tgt):
            loss += self.loss_fn(gen, ref)

        # todo: lecture 2/20 re loss fns. pre-train with teacher forcing, finalize using own predictions

        loss.backward()
        self.optimizer.step()

        # Normalize loss by target length
        return loss.data[0] / tgt_length

    def train(self, train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab,
              num_epochs, max_gen_length=100, checkpoint=20000, debug=False):

        batch_size = 64
        batches = make_batches(train_sents, batch_size)
        num_batches = len(train_sents)

        self.monitor.iters_per_epoch = num_batches
        logger.info("Starting training:")
        self.monitor.start_training()

        total_iters = 0
        
        for ep in range(num_epochs):
            logger.info("Epoch %d:" % ep)
            if not debug:
                random.shuffle(train_sents_vars) #note: should shuffle within batch when batching

            for iteration in range(num_batches):
                src_sent, tgt_sent = pair2var(batches[iteration])

                batch_length = src_sent.size()[1]  # size of longest src sent in batch
                loss = self.train_step(src_sent, tgt_sent, max_length=batch_length)

                self.monitor.finish_iter('train', loss)

                # todo: evaluate function. every X iterations here calculate dev ppl, bleu every epoch at least

                total_iters += 1

                if total_iters % self.monitor.checkpoint == 0:
                    ep_fraction = (iteration + 1) / num_batches
                    dev_output_file = "dev_output_e{0}.{1}.txt".format(ep, ep_fraction)
                    avg_loss, total_loss = self.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, dev_output_file)
                    self.monitor.finish_iter('dev-cp', avg_loss)

            # end of epoch
            # generate output
            dev_output_file = "dev_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, dev_output_file)
            self.monitor.finish_epoch(ep, 'dev', avg_loss, total_loss)

            tst_output_file = "tst_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.generate(tst_sents, src_vocab, tgt_vocab, max_gen_length, tst_output_file)
            self.monitor.finish_epoch(ep, 'test', avg_loss, total_loss)

        # todo: evaluate bleu

        self.monitor.finish_training()

    #todo: generation
    def generate(self, sents, src_vocab, tgt_vocab, max_gen_length, output_file='output.txt'):
        """Generate sentences, and compute the average loss."""

        total_loss = 0.0
        output = []

        for sent in sents:
            src_ref = sent[0]
            tgt_ref = sent[1]
            sent_var = pair2var(sent)
            src_words = [src_vocab.idx2word[i] for i in src_ref]
            tgt_words = [tgt_vocab.idx2word[i] for i in tgt_ref]
            scores, predicted = self.model.generate(sent_var[0], max_gen_length)
            predicted_words = [tgt_vocab.idx2word[i] for i in predicted]
            # logger.info("Predicted:", predicted_words, "  Truth: ", tgt_words)
            output.append(" ".join(predicted_words))
            for gen, ref in zip(scores, sent_var[1]):
                loss = self.loss_fn(gen, ref)
                total_loss += loss.data[0] / len(tgt_ref)

        with open(OUTPUT_PATH + '/' + output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

        avg_loss = total_loss / len(sents)
        return avg_loss, total_loss
