"""training fns"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

#local imports
from utils import time_elapsed, save_plot, use_cuda, pair2var, perplexity
from preprocessing import SOS, SOS_TOKEN, EOS, EOS_TOKEN
from batching import make_batches
import mod_embed as me

import logging
from plot_attention import plot_attention

logger = logging.getLogger(__name__)

def optimizer_factory(optim_type, model, **kwargs):
    assert optim_type in ['SGD', 'Adam'], 'Optimizer type not one of currently supported options'
    return getattr(optim, optim_type)(model.parameters(), **kwargs)


class MTTrainer:

    def __init__(self, model, train_monitor, optim_type='SGD', batch_size=64, beam_size=5, learning_rate=0.01,
                 bpe_incrementer=None):
        self.model = model
        self.optimizer = optimizer_factory(optim_type, model, lr=learning_rate)
        # Decay learning rate by a factor of 0.5 (gamma) every 10 epochs (step_size)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        # Reduce flag makes this return a loss per patch, necessary for masking
        self.loss_fn = nn.NLLLoss(reduce=False)
        self.use_nllloss = True
        self.monitor = train_monitor
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.bpe_incrementer = bpe_incrementer

    # Computes loss for a single batch of source and target sentences
    def calc_batch_loss(self, src, tgt):
        tgt_length = tgt.shape[1]
        decoder_scores = self.model(src, tgt)

        # Compute masks by looking at # of EOS symbols in each sentence, then create masks with (# EOS - 1) 0's at end
        padding = tgt.eq(EOS).sum(1).sub(1)
        masks = Variable(torch.FloatTensor([[[1] * (tgt_length - i) + [0] * i] for i in padding.data])).squeeze(1)
        if use_cuda:
            masks = masks.cuda()

        loss = 0.0
        denom = 0
        if len(masks) != len(tgt) != len(decoder_scores):
            raise Exception
        for gen, ref, mask in zip(decoder_scores, tgt.transpose(0, 1), masks.transpose(0, 1)):
            losses = self.loss_fn(gen, ref)
            # Only average over the non-zero losses from the mask
            losses = losses * mask
            zeros = mask.eq(0).sum()
            denom += losses.shape[0] - zeros.data[0]  # non-masked length
            loss += losses.sum()

        # todo: lecture 2/20 re loss fns. pre-train with teacher forcing, finalize using own predictions

        # normalize by tgt length
        loss = loss / denom
        return loss

    def train_step(self, src, tgt):

        self.optimizer.zero_grad()
        loss = self.calc_batch_loss(src, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0) #gradient clipping
        self.optimizer.step()
        return loss.data[0]

    def train(self, train_sents, dev_sents_sorted, dev_sents_unsorted, tst_sents, src_vocab, tgt_vocab,
              num_epochs, max_gen_length=100, debug=False, output_path="output/"):

        batches = make_batches(train_sents, self.batch_size)
        dev_batches = make_batches(dev_sents_sorted, self.batch_size)
        num_batches = len(batches)

        self.monitor.set_iters(num_batches)
        logger.info("Starting training with %d per batch, %d total batches." % (self.batch_size, num_batches))
        self.monitor.start_training()

        total_iters = 0
                
        #self.calc_dev_loss(dev_batches)

        for ep in range(num_epochs):
            logger.info("Epoch %d:" % ep)
            self.scheduler.step()  #lr scheduler epoch+=1

            if ep % 10 == 0 and ep>0:
                logger.info("Updating learning rate: {}".format(self.scheduler.get_lr()))

            if not debug:
                random.shuffle(batches)

            for iteration in range(num_batches):
                if use_cuda:
                    torch.cuda.empty_cache()
                src_sent, tgt_sent = pair2var(batches[iteration])

                max_batch_length = src_sent.size()[1]  # size of longest src sent in batch
                loss = self.train_step(src_sent, tgt_sent)

                self.monitor.finish_iter('train', loss)
                total_iters += 1

                if total_iters % self.monitor.checkpoint == 0:
                    logger.info("Calculating dev loss + writing output")
                    ep_fraction = (iteration + 1) / num_batches
                    dev_output_file = "dev_output_e{0}.{1}.txt".format(ep, ep_fraction)
                    avg_loss, total_loss = self.generate(dev_sents_unsorted, src_vocab, tgt_vocab,
                                                         max_gen_length, dev_output_file, output_path)
                    self.monitor.finish_iter('dev-cp', avg_loss)

            # end of epoch
            # generate output
            logger.info("Calculating dev loss + writing output")
            dev_output_file = "dev_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.calc_dev_loss(dev_batches)
            self.generate(dev_sents_unsorted, src_vocab, tgt_vocab, max_gen_length, dev_output_file, output_path)
            done_training = self.monitor.finish_epoch(ep, 'dev', avg_loss, total_loss)
            if done_training:
                logger.info("Stopping early: dev loss has not gone down in patience period.")
                break

            # todo: evaluate bleu

            tst_output_file = "tst_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.generate(tst_sents, src_vocab, tgt_vocab, max_gen_length, tst_output_file, output_path)
            self.monitor.finish_epoch(ep, 'test', avg_loss, total_loss)

            # todo: check threshold for incremental bpe
            if self.bpe_incrementer and True:
                self.bpe_incrementer.update_bpe_vocab(self.model, self.optimizer, tgt_vocab)
                train_sents, dev_sents_sorted, dev_sents_unsorted, tst_sents = self.bpe_incrementer\
                    .load_next_bpe(src_vocab, tgt_vocab)
                batches = make_batches(train_sents, self.batch_size)
                dev_batches = make_batches(dev_sents_sorted, self.batch_size)
                num_batches = len(batches)

                self.monitor.set_iters(num_batches)

        #-----------------------------
        self.monitor.finish_training()


    def calc_dev_loss(self, dev_batches):
        total_loss = 0.0
        num_processed = 0
        sent_id = 0
        num_batches = len(dev_batches)
        for iteration in range(num_batches):
            if iteration % 50 == 0 and num_batches > 50:
                logger.info("{}/{} dev batches complete".format(iteration, num_batches))
            src, tgt = pair2var(dev_batches[iteration], volatile=True)
            loss = self.calc_batch_loss(src, tgt)
            total_loss += loss
            # todo: This is not the ideal way to clear the computation graph since it does the full backprop, but doesn't update parameters.
#            loss.backward()
        avg_loss = total_loss / len(dev_batches)
        return avg_loss.data[0], total_loss.data[0]

    #todo: generation
    def generate(self, sents, src_vocab, tgt_vocab, max_gen_length, output_file='output.txt', output_path="output/", plot_attn=False):
        """Generate sentences, and compute the average loss."""

        total_loss = 0.0
        output = []
        num_processed = 0
        sent_id = 0
        for sent in sents:
            sent_id += 1
            src_ref = sent[0]
            tgt_ref = sent[1]
            sent_var = pair2var(sent)
            src_words = [src_vocab.idx2word[i] for i in src_ref]
            scores, predicted, attention = self.model.generate(sent_var[0].view(1, len(src_words)),
                                                                   max_gen_length, self.beam_size)
            # print(attention)
            predicted_words = [tgt_vocab.idx2word[i] for i in predicted]
            src_words = [src_vocab.idx2word[i] for i in src_ref]
            if plot_attn and attention is not None:
                plot_attention(src_words, predicted_words, attention.data.cpu().numpy(), output_path + str(sent_id) + '.png')

            if predicted_words[0] == SOS_TOKEN:
                predicted_words = predicted_words[1:]

            if EOS_TOKEN in predicted_words:
                eos_index = predicted_words.index(EOS_TOKEN)
                predicted_words = predicted_words[:eos_index]
            output.append(" ".join(predicted_words))
            gen_ref_pairs = list(zip(scores, sent_var[1]))
            for gen, ref in gen_ref_pairs:
                loss = self.loss_fn(gen, ref).mean()
                total_loss += loss.data[0] / len(gen_ref_pairs)
            num_processed += 1
            if num_processed % 100 == 0:
                logger.info("Processed {} sentences.".format(num_processed))


        with open(output_path + '/' + output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

        avg_loss = total_loss / len(sents)
        return avg_loss, total_loss
