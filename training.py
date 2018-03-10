"""training fns"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#local imports
from utils import time_elapsed, save_plot, use_cuda, pair2var, perplexity, MODEL_PATH, OUTPUT_PATH
from preprocessing import SOS, EOS, EOS_TOKEN
from batching import make_batches

import logging

logger = logging.getLogger(__name__)

def optimizer_factory(optim_type, model, **kwargs):
    assert optim_type in ['SGD', 'Adam'], 'Optimizer type not one of currently supported options'
    return getattr(optim, optim_type)(model.parameters(), **kwargs)


class MTTrainer:

    def __init__(self, model, train_monitor, optim_type='SGD', batch_size=64, learning_rate=0.01):
        self.model = model
        self.optimizer = optimizer_factory(optim_type, model, lr=learning_rate)
        # Reduce flag makes this return a loss per patch, necessary for masking
        self.loss_fn = nn.NLLLoss(reduce=False)
        self.use_nllloss = True
        self.monitor = train_monitor
        self.batch_size = batch_size

    def train_step(self, src, tgt, max_length):

        self.optimizer.zero_grad()
        loss = 0.0
        # Dimensions are (batch_size, sequence_length)
        tgt_length = tgt.shape[1]
        decoder_scores = self.model(src, tgt)

        # Compute masks by looking at # of EOS symbols in each sentence, then create masks with (# EOS - 1) 0's at end
        padding = tgt.eq(EOS).sum(1).sub(1)
        masks = Variable(torch.FloatTensor([[[1] * (tgt_length - i) + [0] * i] for i in padding.data])).squeeze(1)
        if use_cuda:
            masks = masks.cuda()

        for gen, ref, mask in zip(decoder_scores, tgt.transpose(0, 1), masks.transpose(0, 1)):
            losses = self.loss_fn(gen, ref)
            losses = losses * mask
            # Only average over the non-zero losses from the mask
            zeros = mask.eq(0).sum()
            denom = losses.shape[0] - zeros.data[0]
            avg_loss = losses.sum() / denom
            loss += avg_loss

        # todo: lecture 2/20 re loss fns. pre-train with teacher forcing, finalize using own predictions

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0) #gradient clipping
        self.optimizer.step()

        # Normalize loss by target length
        return loss.data[0] / tgt_length

    def train(self, train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab,
              num_epochs, max_gen_length=100, debug=False):

        batches = make_batches(train_sents, self.batch_size)
        num_batches = len(batches)

        self.monitor.set_iters(num_batches)
        logger.info("Starting training with %d per batch." % self.batch_size)
        self.monitor.start_training()

        total_iters = 0
        
        for ep in range(num_epochs):
            logger.info("Epoch %d:" % ep)
            if not debug:
                random.shuffle(batches) #note: should shuffle within batch when batching

            for iteration in range(num_batches):
                src_sent, tgt_sent = pair2var(batches[iteration])

                max_batch_length = src_sent.size()[1]  # size of longest src sent in batch
                loss = self.train_step(src_sent, tgt_sent, max_length=max_batch_length)

                self.monitor.finish_iter('train', loss)

                # todo: evaluate function. every X iterations here calculate dev ppl, bleu every epoch at least

                total_iters += 1

                if total_iters % self.monitor.checkpoint == 0:
                    logger.info("Calculating dev loss")
                    ep_fraction = (iteration + 1) / num_batches
                    dev_output_file = "dev_output_e{0}.{1}.txt".format(ep, ep_fraction)
                    avg_loss, total_loss = self.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, dev_output_file)
                    self.monitor.finish_iter('dev-cp', avg_loss)

            # end of epoch
            # generate output
            logger.info("Calculating dev loss")
            dev_output_file = "dev_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, dev_output_file)
            self.monitor.finish_epoch(ep, 'dev', avg_loss, total_loss)

        # todo: evaluate bleu

        self.monitor.finish_training()

        tst_output_file = "tst_output_e{0}.txt".format(ep)
        avg_loss, total_loss = self.generate(tst_sents, src_vocab, tgt_vocab, max_gen_length, tst_output_file)
        self.monitor.finish_epoch(ep, 'test', avg_loss, total_loss)



    #todo: generation
    def generate(self, sents, src_vocab, tgt_vocab, max_gen_length, output_file='output.txt'):
        """Generate sentences, and compute the average loss."""

        total_loss = 0.0
        output = []
        num_processed = 0
        for sent in sents:
            src_ref = sent[0]
            tgt_ref = sent[1]
            sent_var = pair2var(sent)
            src_words = [src_vocab.idx2word[i] for i in src_ref]
            scores, predicted = self.model.generate(sent_var[0].view(1, len(src_words)), max_gen_length)
            predicted_words = [tgt_vocab.idx2word[i] for i in predicted]
            if EOS_TOKEN in predicted_words:
                eos_index = predicted_words.index(EOS_TOKEN) + 1
                predicted_words = predicted_words[:eos_index]
            # logger.info("Predicted:", predicted_words, "  Truth: ", tgt_words)
            output.append(" ".join(predicted_words))
            for gen, ref in zip(scores, sent_var[1]):
                loss = self.loss_fn(gen, ref).mean()
                total_loss += loss.data[0] / len(tgt_ref)
            num_processed += 1
            if num_processed % 100 == 0:
                print("Processed {} sentences.".format(num_processed))


        with open(OUTPUT_PATH + '/' + output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

        avg_loss = total_loss / len(sents)
        return avg_loss, total_loss
