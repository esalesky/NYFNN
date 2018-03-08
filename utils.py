import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
OUTPUT_PATH = 'output/'
MODEL_PATH = 'models/'


def pair2var(sent_pair):
    src_variable = Variable(torch.LongTensor(sent_pair[0]))
    tgt_variable = Variable(torch.LongTensor(sent_pair[1]))
    if use_cuda:
        return (src_variable.cuda(), tgt_variable.cuda())
    else:
        return (src_variable, tgt_variable)


def format_time(s):
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)


def time_elapsed(start, percent):  #todo: make this realistic plz
    now = time.time()
    s = now - start
    es = s / (percent)
    rs = es - s
    return '%s (est. remaining in epoch: %s)' % (format_time(s), format_time(rs))


def save_plot(points, name, freq):
    #plt.figure()
    fig, ax = plt.subplots()
    # puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=10.0)
    # ax.yaxis.set_major_locator(loc)
    x_vals = np.arange(0, freq*len(points), freq)
    plt.plot(x_vals, points)
    plt.xlabel('Iteration')
    plt.ylabel(name.title())
    plt.title(name.title())
    plt.savefig('{}{}.jpg'.format(OUTPUT_PATH, name))
    plt.close(fig)


def perplexity(loss):
    """Prints the perplexity given the NLLLoss."""
    max_loss = 100
    if loss > max_loss:
        return math.exp(max_loss)
    else:
        return math.exp(loss)
