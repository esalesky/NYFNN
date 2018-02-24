import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import torch
from torch.autograd import Variable

use_cuda = False #torch.cuda.is_available()
OUTPUT_PATH = 'output/'
MODEL_PATH = 'models/'


def pair2var(sent_pair):
    src_variable = Variable(torch.LongTensor(sent_pair[0])).view(-1,1)
    tgt_variable = Variable(torch.LongTensor(sent_pair[1])).view(-1,1)
    if use_cuda:
        return (src_variable.cuda(), tgt_variable.cuda())
    else:
        return (src_variable, tgt_variable)


def min_sec(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_elapsed(start, percent):
    now = time.time()
    s = now - start
    es = s / (percent)
    rs = es - s
    return '%s (est. remaining: %s)' % (min_sec(s), min_sec(rs))


def save_plot(points, name, freq):
    plt.figure()
    fig, ax = plt.subplots()
    # puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=1.0)
    ax.yaxis.set_major_locator(loc)
    x_vals = np.arange(0, freq*len(points), freq)
    plt.plot(x_vals, points)
    plt.xlabel('Iteration')
    plt.ylabel(name.title())
    plt.title(name.title())
    plt.savefig('{}{}.jpg'.format(OUTPUT_PATH, name))


def perplexity(loss):
    """Prints the perplexity given the NLLLoss."""
    max_loss = 100
    if loss > max_loss:
        return math.exp(max_loss)
    else:
        return math.exp(loss)