"""training fns"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#local imports
from utils import time_elapsed, save_plot, use_cuda, pair2var, perplexity, MODEL_PATH
from preprocessing import SOS, EOS


def train(src, tgt, model, optimizer, loss_fn, max_length):

    optimizer.zero_grad()
    loss = 0.0
    tgt_length = tgt.size()[0]

    decoder_scores, words = model(src, tgt)
    
    for gen, ref in zip(decoder_scores, tgt):
        # print("Gen: ", gen, "Ref: ", ref)
        loss += loss_fn(gen, ref)

    #todo: lecture 2/20 re loss fns. pre-train with teacher forcing, finalize using own predictions

    loss.backward()
    optimizer.step()

    # Normalize loss by target length
    return loss.data[0] / tgt_length


#todo: generation
def generate(model, sents, src_vocab, tgt_vocab, max_gen_length, loss_fn, output_file='output.txt'):
    """Generate sentences, and compute the average loss."""

    total_loss = 0.0
    output = []

    for sent in sents:
        src_ref = sent[0]
        tgt_ref = sent[1]
        sent_var = pair2var(sent)
        src_words = [src_vocab.idx2word[i] for i in src_ref]
        tgt_words = [tgt_vocab.idx2word[i] for i in tgt_ref]
        scores, predicted = model.generate(sent_var[0], max_gen_length)
        predicted_words = [tgt_vocab.idx2word[i] for i in predicted]
#        print("Predicted:", predicted_words, "  Truth: ", tgt_words)
        output.append(" ".join(predicted_words))
        for gen, ref in zip(scores, sent_var[1]):
            loss = loss_fn(gen, ref)
            total_loss += loss.data[0] / len(tgt_ref)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output))

    avg_loss = total_loss / len(sents)
    return avg_loss, total_loss


def train_setup(model, train_sents, dev_sents, tst_sents, src_vocab, tgt_vocab, num_epochs,
                learning_rate=0.01, print_every=1000, plot_every=100, model_every=20000, max_gen_length=100):
    start = time.time()
    train_plot_losses = []
    dev_plot_losses   = []
    train_plot_perplexities = []
    dev_plot_perplexities   = []
    print_perplexity_avg = -1
    print_loss_total = 0  #resets every print_every
    plot_loss_total  = 0  #resets every plot_every

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_sents_vars = [ pair2var(s) for s in train_sents ]
    # Use NLLLoss
    loss_fn = nn.NLLLoss()
    use_nllloss = True

    num_batches = len(train_sents)  #todo: currently batch_size=1 every sentence is a batch

    print("Starting training:")
    for ep in range(num_epochs):
        print("Epoch %d:" % ep)
        random.shuffle(train_sents_vars)
        
        for iteration in range(num_batches):
            src_sent = train_sents_vars[iteration][0]
            tgt_sent = train_sents_vars[iteration][1]

            batch_length = src_sent.size()[0]  #size of longest src sent in batch
            loss = train(src_sent, tgt_sent, model, optimizer, loss_fn, max_length=batch_length)
            print_loss_total += loss
            plot_loss_total  += loss
    
            #todo: evaluate function. every X iterations here calculate dev ppl, bleu every epoch at least
            
            # log
            if (iteration + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                perc_through_epoch = (iteration + 1) / num_batches
                print('Iter: {} / {}.  {}'.format(iteration + 1, num_batches, time_elapsed(start, perc_through_epoch)))
                print('\tLoss: {0:.4f}'.format(print_loss_avg))
                if use_nllloss:
                    print_perplexity_avg = perplexity(print_loss_avg)
                    print('\tPerplexity: {0:.4f}'.format(print_perplexity_avg))

            # append losses for plot
            if (iteration + 1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                train_plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                if use_nllloss:
                    plot_perplexity_avg = perplexity(plot_loss_avg)
                    train_plot_perplexities.append(plot_perplexity_avg)

            if iteration % model_every == 0 and iteration > 0:
                # save the model
                ep_fraction = int((iteration+1)/model_every)
                if use_nllloss:
                    model_name = "model_e{0}.{1}_perp{2:.3f}".format(ep, ep_fraction, print_perplexity_avg)
                else:
                    model_name = "model_e{0}.{1}_loss{2:.3f}".format(ep, ep_fraction, print_perplexity_avg)
                model.save("{}{}.pkl".format(MODEL_PATH, model_name))
                # generate output
                dev_output_file = "dev_output_e{0}.{1}.txt".format(ep, ep_fraction)
                avg_loss, total_loss = generate(model, dev_sents, src_vocab, tgt_vocab, max_gen_length, loss_fn, dev_output_file)
                dev_ppl = perplexity(avg_loss)
                dev_plot_losses.append(avg_loss)
                dev_plot_perplexities.append(dev_ppl)
                print("-"*65)
                print("Epoch {}: dev ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(ep, dev_ppl, avg_loss, total_loss))
                print("-"*65)

                save_plot(dev_plot_losses, 'dev_loss', 1)
                save_plot(dev_plot_perplexities, 'dev_perplexity', 1)
                save_plot(train_plot_losses, 'train_loss', plot_every)
                save_plot(train_plot_perplexities, 'train_perplexity', plot_every)
                

        # end of epoch
        # generate output
        dev_output_file = "dev_output_e{0}.txt".format(ep)
        avg_loss, total_loss = generate(model, dev_sents, src_vocab, tgt_vocab, max_gen_length, loss_fn, dev_output_file)
        
        dev_ppl = perplexity(avg_loss)
        dev_plot_losses.append(avg_loss)
        dev_plot_perplexities.append(dev_ppl)
        print("-"*65)
        print("Epoch {}: dev ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(ep, dev_ppl, avg_loss, total_loss))
        print("-"*65)

        tst_output_file = "tst_output_e{0}.txt".format(ep)
        avg_loss, total_loss = generate(model, tst_sents, src_vocab, tgt_vocab, max_gen_length, loss_fn, 'tst_output.txt')
        tst_ppl = perplexity(avg_loss)
        print("-"*65)
        print("Epoch {}: tst_ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(ep, tst_ppl, avg_loss, total_loss))
        print("-"*65)

    #todo: evaluate bleu

    save_plot(train_plot_losses, 'train_loss', plot_every)
    save_plot(train_plot_perplexities, 'train_perplexity', plot_every)

    save_plot(dev_plot_losses, 'dev_loss', 1)
    save_plot(dev_plot_perplexities, 'dev_perplexity', 1)
