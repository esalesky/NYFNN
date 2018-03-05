"""training fns"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#local imports
from utils import time_elapsed, save_plot, use_cuda, pair2var, perplexity, MODEL_PATH, OUTPUT_PATH
from preprocessing import SOS, EOS

import logging

logger = logging.getLogger(__name__)

def optimizer_factory(optim_type, model, **kwargs):
    assert optim_type in ['SGD', 'Adam'], 'Optimizer type not one of currently supported options'
    return getattr(optim, optim_type)(model.parameters(), **kwargs)


class MTTrainer:

    def __init__(self, model, optim_type='SGD', learning_rate=0.01):
        self.model = model
        self.optimizer = optimizer_factory(optim_type, model, lr=learning_rate)
        self.loss_fn = nn.NLLLoss()
        self.use_nllloss = True

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

    def train(self, train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab, num_epochs,
                    print_every=1000, plot_every=100, model_every=20000, max_gen_length=100):
        start = time.time()
        train_plot_losses = []
        dev_plot_losses = []
        train_plot_perplexities = []
        dev_plot_perplexities = []
        print_perplexity_avg = -1
        print_loss_total = 0  # resets every print_every
        plot_loss_total = 0  # resets every plot_every

        train_sents_vars = [pair2var(s) for s in train_sents]
        # Use NLLLoss

        num_batches = len(train_sents)  # todo: currently batch_size=1 every sentence is a batch

        logger.info("Starting training:")
        for ep in range(num_epochs):
            logger.info("Epoch %d:" % ep)
            random.shuffle(train_sents_vars)

            for iteration in range(num_batches):
                src_sent = train_sents_vars[iteration][0]
                tgt_sent = train_sents_vars[iteration][1]

                batch_length = src_sent.size()[0]  # size of longest src sent in batch
                loss = self.train_step(src_sent, tgt_sent, max_length=batch_length)
                print_loss_total += loss
                plot_loss_total += loss

                # todo: evaluate function. every X iterations here calculate dev ppl, bleu every epoch at least

                # log
                if (iteration + 1) % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    perc_through_epoch = (iteration + 1) / num_batches
                    logger.info('Iter: {} / {}.  {}'.format(iteration + 1, num_batches,
                                                      time_elapsed(start, perc_through_epoch)))
                    logger.info('\tLoss: {0:.4f}'.format(print_loss_avg))
                    if self.use_nllloss:
                        print_perplexity_avg = perplexity(print_loss_avg)
                        logger.info('\tPerplexity: {0:.4f}'.format(print_perplexity_avg))

                # append losses for plot
                if (iteration + 1) % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    train_plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
                    if self.use_nllloss:
                        plot_perplexity_avg = perplexity(plot_loss_avg)
                        train_plot_perplexities.append(plot_perplexity_avg)

                if iteration % model_every == 0 and iteration > 0:
                    # save the model
                    ep_fraction = int((iteration + 1) / model_every)
                    if self.use_nllloss:
                        model_name = "model_e{0}.{1}_perp{2:.3f}".format(ep, ep_fraction, print_perplexity_avg)
                    else:
                        model_name = "model_e{0}.{1}_loss{2:.3f}".format(ep, ep_fraction, print_perplexity_avg)
                    self.model.save("{}{}.pkl".format(MODEL_PATH, model_name))
                    # generate output
                    dev_output_file = "dev_output_e{0}.{1}.txt".format(ep, ep_fraction)
                    avg_loss, total_loss = self.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, dev_output_file)
                    dev_ppl = perplexity(avg_loss)
                    dev_plot_losses.append(avg_loss)
                    dev_plot_perplexities.append(dev_ppl)
                    logger.info("-" * 65)
                    logger.info(
                        "Epoch {}: dev ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(ep, dev_ppl, avg_loss,
                                                                                                 total_loss))
                    logger.info("-" * 65)

                    save_plot(dev_plot_losses, 'dev_loss', 1)
                    save_plot(dev_plot_perplexities, 'dev_perplexity', 1)
                    save_plot(train_plot_losses, 'train_loss', plot_every)
                    save_plot(train_plot_perplexities, 'train_perplexity', plot_every)

            # end of epoch
            # generate output
            dev_output_file = "dev_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.generate(dev_sents, src_vocab, tgt_vocab, max_gen_length, dev_output_file)

            dev_ppl = perplexity(avg_loss)
            dev_plot_losses.append(avg_loss)
            dev_plot_perplexities.append(dev_ppl)
            logger.info("-" * 65)
            logger.info("Epoch {}: dev ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(ep, dev_ppl, avg_loss,
                                                                                           total_loss))
            logger.info("-" * 65)

            tst_output_file = "tst_output_e{0}.txt".format(ep)
            avg_loss, total_loss = self.generate(tst_sents, src_vocab, tgt_vocab, max_gen_length, 'tst_output.txt')
            tst_ppl = perplexity(avg_loss)
            logger.info("-" * 65)
            logger.info("Epoch {}: tst_ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(ep, tst_ppl, avg_loss,
                                                                                           total_loss))
            logger.info("-" * 65)

        # todo: evaluate bleu

        save_plot(train_plot_losses, 'train_loss', plot_every)
        save_plot(train_plot_perplexities, 'train_perplexity', plot_every)

        save_plot(dev_plot_losses, 'dev_loss', 1)
        save_plot(dev_plot_perplexities, 'dev_perplexity', 1)

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



