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


def save_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('loss_plot.jpg')

