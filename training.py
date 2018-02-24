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

    decoder_output = model(src, tgt)
    
    for gen, ref in zip(decoder_output, tgt):
        # print("Gen: ", gen, "Ref: ", ref)
        loss += loss_fn(gen, ref)

    #todo: lecture 2/20 re loss fns. pre-train with teacher forcing, finalize using own predictions

    loss.backward()
    optimizer.step()

    # Normalize loss by target length
    return loss.data[0] / tgt_length


#todo: generation
def generate(model, sents, src_vocab, tgt_vocab, max_gen_length, output_file='output.txt'):
    """Generate sentences, and compute the average loss."""

    total_loss = 0.0
    output = []

    for sent in sents:
        src_ref = sent[0]
        tgt_ref = sent[1]
        src_words = [src_vocab.idx2word[i] for i in src_ref]
        tgt_words = [tgt_vocab.idx2word[i] for i in tgt_ref]
        output.append(" ".join(src_words))
        predicted = model.generate(pair2var(sent)[0], max_gen_length)
        predicted_words = [tgt_vocab.idx2word[i] for i in predicted]
        print("Predicted:", predicted_words, "Truth: ", tgt_words)

        for gen, ref in zip(predicted, tgt_ref):
            total_loss += loss_fn(gen, ref) / len(sent)

    with open(output_file, 'w') as f:
        f.write("\n".join(output))

    avg_loss = total_loss / len(sents)
    return avg_loss, total_loss


def train_setup(model, train_sents, dev_sents, tst_sents, num_epochs, learning_rate=0.01,
                print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  #resets every print_every
    plot_loss_total  = 0  #resets every plot_every

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_sents_vars = [ pair2var(s) for s in train_sents ]
    # Use NLLLoss
    loss_fn = nn.NLLLoss()
    use_nllloss = True
    plot_perplexities = []

    num_batches = len(sents)  #todo: currently batch_size=1 every sentence is a batch

    print("Starting training:")
    for ep in range(num_epochs):
        print("Epoch %d:" % ep)
        random.shuffle(sents)
        
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
                perc_through_training = (ep + iteration / num_batches) / num_epochs
                print('Iter: {} / {}.  {}'.format(iteration + 1, num_batches, time_elapsed(start, perc_through_training)))
                print('\tLoss: {0:.4f}'.format(print_loss_avg))
                if use_nllloss:
                    print_perplexity_avg = perplexity(print_loss_avg)
                    print('\tPerplexity: {0:.4f}'.format(print_perplexity_avg))

            # append losses for plot
            if (iteration + 1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                if use_nllloss:
                    plot_perplexity_avg = perplexity(plot_loss_avg)
                    plot_perplexities.append(plot_perplexity_avg)

        # end of epoch
        # save the model
        if use_nllloss:
            model_name = "model_e{0}_perp{1:.3f}".format(ep, print_perplexity_avg)
        else:
            model_name = "model_e{0}_loss{1:.3f}".format(ep, print_perplexity_avg)
        model.save("{}{}.pkl".format(MODEL_PATH, model_name))
        # generate output
        avg_loss, total_loss = generate(model, dev_sents, src_vocab, tgt_vocab, max_sent_length, "dev_output.txt")
        dev_ppl = perplexity(avg_loss)
        print("-"*20+"\n")
        print("Epoch %d: dev ppl, %f. avg loss, %f. total loss, %f" % dev_ppl, avg_loss, total_loss)
        print("-"*20+"\n")

    # end of training
    avg_loss, total_loss = generate(model, test_sents, src_vocab, tgt_vocab, max_sent_length, 'tst_output.txt')
    tst_ppl = perplexity(avg_loss)
    print("-"*20+"\n")
    print("Epoch %d: tst_ppl, %f. avg loss, %f. total loss, %f" % tst_ppl, avg_loss, total_loss)
    print("-"*20+"\n")

    #todo: evaluate bleu

    save_plot(plot_losses, 'loss', plot_every)
    save_plot(plot_perplexities, 'perplexity', plot_every)
