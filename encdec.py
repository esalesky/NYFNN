import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from utils import use_cuda


#todo: minibatching

def rnn_factory(rnn_type, **kwargs):
    assert rnn_type in ['LSTM','GRU'], 'rnn_type not one of currently supported options'
    rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn

class RNNEncoder(nn.Module):
    """simple initial encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.vocab_size  = vocab_size  #source vocab size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_size)  #currently embed_size = hidden_size
        self.rnn = rnn_factory(rnn_type, input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)  #requires embed_size = hidden_size
        self.hidden = self.init_hidden()

    #src is currently single sentence
    def forward(self, src):
        embedded = self.embedding(src)
        output, self.hidden = self.rnn(embedded.view(len(src), 1, -1), self.hidden)
        return output, self.hidden

    def init_hidden(self):
        #dimensions: num_layers, minibatch_size, hidden_dim
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            self.hidden = result.cuda()
        else:
            self.hidden = result

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class RNNDecoder(nn.Module):
    """simple initial decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM',  num_layers=1, bidirectional=False):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size  #target vocab size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn_factory(rnn_type, input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)  #requires embed_size = hidden_size
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  #dim corresponding to vocab
        self.hidden = self.init_hidden()

    def forward(self, tgt):
        #teacher forcing: use target as next input
        output = self.embedding(tgt).view(len(tgt), 1, -1)
        output = F.relu(output)
        output, self.hidden = self.rnn(output, self.hidden)
        intermediate = self.out(output)  #maps to len(sent) x 1 x vocab_size
        output = self.softmax(intermediate) #softmax across vocab per word
        return output

    def init_hidden(self):
        #dimensions: num_layers, minibatch_size, hidden_dim
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class EncDec(nn.Module):
    """initial encoder + decoder model"""
    def __init__(self, encoder, decoder):
        super(EncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    #src,tgt currently single sentences
    def forward(self, src, tgt, batch_len):
        self.encoder.hidden = self.encoder.init_hidden()
        encoder_output = Variable(torch.zeros(batch_len, self.encoder.hidden_size))
        encoder_output = self.encoder(src)
        self.decoder.hidden = self.encoder.hidden
        output = self.decoder(tgt)
        
        return output

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
